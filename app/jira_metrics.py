#!/usr/bin/env python

import confuse
from jira import JIRA
import datetime as dt
import math
import pandas as pd
from dateutil.relativedelta import relativedelta
import uuid
import os


cfg = confuse.Configuration('JiraMetrics', __name__)


def atlassian_auth(override_config_filename=None):
    """Authenticate on the Jira Cloud instance"""

    if override_config_filename is not None:
        cfg.set_file(override_config_filename)

    username = cfg['Connection']['Username'].get()
    api = cfg['Connection']['ApiKey'].get()

    jira = JIRA(
        server=cfg['Connection']['Domain'].get(),
        basic_auth=(username, api)
    )
    return jira


def jql_search(jira_obj, jql_query=None):
    """Run a JQL search and return the jira object with results"""
    sfields = [
        "created",
        "issuetype"
    ]
    if jql_query is None:
        jql_query = cfg['Query'].get()

    issues = jira_obj.search_issues(
        jql_query,
        fields=sfields,
        maxResults=99999,
        expand='changelog'
    )
    return issues


def convert_cfd_table(issues_obj):
    """Convert the issues obj into a dictionary on the cfd format"""
    cfd_table = []
    for issue in issues_obj:
        # start creating our line of the table with field: value
        cfd_line = {}
        cfd_line["issue"] = issue.key
        cfd_line["issuetype"] = group_issuetype(issue.fields.issuetype.name)
        cfd_line["cycletime"] = 0
        cfd_line["final_datetime"] = 0

        # create other columns according to workflow in cfg
        workflows = cfg['Workflow'].get()
        for key, value in workflows.items():
            cfd_line[key.lower()] = 0

        # store final status
        fstatus = list(cfd_line.keys())[-1]

        # create a mini dict to organize itens
        # (issue_time, history_time, from_status, to_status)
        status_table = []
        for history in issue.changelog.histories:
            for item in history.items:
                # only items that are status change are important to us
                if item.field == 'status':
                    status_line = {}
                    status_line["history_datetime"] = history.created
                    status_line["from_status"] = group_status(item.fromString)
                    status_line["to_status"] = group_status(item.toString)
                    status_table.append(status_line)
                    # store in finaldatetime the highest timestamp to fix
                    # items with many 'done' transitions
                    stamp_created = convert_jira_datetime(history.created)
                    if (group_status(item.toString) == fstatus and
                            stamp_created > cfd_line["final_datetime"]):
                        cfd_line["final_datetime"] = stamp_created
        status_table.reverse()
        # send the mini dict to be processed and return the workflow times
        cfd_line = process_status_table(status_table, cfd_line)
        # special case: time on the first status should be compared to when
        # the issue was created it is always the first line of the status table
        cfd_line[status_table[0]['from_status']] += calc_diff_date_to_unix(
            issue.fields.created,
            status_table[0]['history_datetime']
        )
        # add line to table
        cfd_table.append(cfd_line)

    return cfd_table


def process_status_table(status_table, cfd_line):
    from_table = []
    from_table.extend(status_table)
    to_table = []
    to_table.extend(status_table)

    # everytime that I have fromString(enddatetime)
    # I should find a toString(startdatetime)
    for from_item in from_table:
        to_item_index = 0
        for to_item in to_table:
            if to_item['to_status'] == from_item['from_status']:
                # send to calc
                # add the time to the column corresponding the enddatetime
                cfd_line[from_item['from_status']] += calc_diff_date_to_unix(
                    to_item['history_datetime'], from_item['history_datetime'])
                del to_table[to_item_index]
                break
            to_item_index += 1

    return cfd_line


def group_issuetype(issuetype):
    types = cfg['Issuetype'].get()
    for key1, value1 in types.items():
        if type(value1) == list:
            for value2 in types[key1]:
                if value2 == issuetype:
                    return key1
        else:
            if value1 == issuetype:
                return key1
    raise Exception(
        "Can't find issuetype in config file: {}".format(issuetype))


def group_status(status):
    workflows = cfg['Workflow'].get()
    for key1, value1 in workflows.items():
        if type(value1) == list:
            for value2 in workflows[key1]:
                if value2.lower() == status.lower():
                    return key1.lower()
        else:
            if value1.lower() == status.lower():
                return key1.lower()

    raise Exception(
        "Can't find status in config file: {}".format(status))


def calc_diff_date_to_unix(start_datetime, end_datetime):
    """Given the start and end datetime
    return the difference in unix timestamp format"""
    start = convert_jira_datetime(start_datetime)
    end = convert_jira_datetime(end_datetime)
    timedelta = end - start
    minutes = math.ceil(timedelta/60)
    return minutes


def convert_jira_datetime(datetime_str):
    """Convert Jira datetime format to unix timestamp"""
    time = dt.datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f%z')
    return dt.datetime.timestamp(time)


def read_dates(dictio):
    kanban_data = pd.DataFrame.from_dict(dictio)
    if kanban_data.empty is False:
        kanban_data.final_datetime = pd.to_datetime(
            kanban_data.final_datetime, unit='s'
        ).dt.date
        # Calculate each issue cycletime
        status_list_lower = [
            v.lower() for v in cfg['Cycletime']['Status'].get()]
        kanban_data.cycletime = kanban_data[status_list_lower].sum(axis=1)
        # Remove items with cycletime == 0
        kanban_data = kanban_data[kanban_data.cycletime != 0]
    return kanban_data


def calc_cycletime_percentile(kanban_data, percentile=None):
    """Calculate cycletime percentiles on cfg with all dict entries"""
    if kanban_data.empty is False:
        if percentile is not None:
            issuetype = kanban_data.groupby('issuetype').cycletime.quantile(
                percentile / 100)
            issuetype['Total'] = kanban_data.cycletime.quantile(
                percentile / 100)
            return issuetype.div(60).div(24)
        else:
            for cfg_percentile in cfg['Cycletime']['Percentiles'].get():
                cycletime = kanban_data.groupby(
                    'issuetype').cycletime.quantile(
                    cfg_percentile / 100)
                cycletime['Total'] = kanban_data.cycletime.quantile(
                    cfg_percentile / 100)
                cycletime = cycletime.div(60).div(24)


def calc_throughput(kanban_data, start_date=None, end_date=None):
    """Change the pandas DF to a Troughput per day format"""
    if start_date is not None and 'final_datetime' in kanban_data.columns:
        kanban_data = kanban_data[~(
            kanban_data['final_datetime'] < start_date)]
    if end_date is not None and 'final_datetime' in kanban_data.columns:
        kanban_data = kanban_data[~(
            kanban_data['final_datetime'] > end_date)]
    if kanban_data.empty is False:
        # Reorganize DataFrame
        throughput = pd.crosstab(
            kanban_data.final_datetime, kanban_data.issuetype, colnames=[None]
        ).reset_index()
        # Sum Throughput per day
        throughput['Throughput'] = 0
        if 'Story' in throughput:
            throughput['Throughput'] += throughput.Story
        if 'Bug' in throughput:
            throughput['Throughput'] += throughput.Bug
        if 'Task' in throughput:
            throughput['Throughput'] += throughput.Task
        if throughput.empty is False:
            date_range = pd.date_range(
                start=throughput.final_datetime.min(),
                end=throughput.final_datetime.max()
            )
            throughput = throughput.set_index(
                'final_datetime'
            ).reindex(date_range).fillna(0).astype(int).rename_axis('Date')

        # Fill all missing dates
        date_range = pd.date_range(start_date, end_date)
        throughput = throughput.reindex(date_range, fill_value=0)
        return throughput


def simulate_montecarlo(throughput, sources=None, simul=None, simul_days=None):
    """
    Simulate Monte Carlo

    Parameters
    ----------
        throughput : dataFrame
            throughput base values of the simulation
        sources : dictionary
            sources that the simulations should run on
        simul : integer
            number of simulations
        simul_days : integer
            days to run the simulation
    """
    if sources is None:
        sources = cfg['Montecarlo']['Source'].get()
    if simul is None:
        simul = cfg['Montecarlo']['Simulations'].get()
    if simul_days is None:
        simul_days = calc_simul_days()

    mc = {}
    for source in sources:
        mc[source] = run_simulation(throughput, source, simul, simul_days)
    return mc


def run_simulation(throughput, source, simul, simul_days):
    """Run monte carlo simulation with the result of how many itens will
    be delivered in a set of days """

    if (throughput is not None and source in throughput.columns):

        dataset = throughput[[source]].reset_index(drop=True)

        samples = [getattr(dataset.sample(
            n=simul_days, replace=True
        ).sum(), source) for i in range(simul)]

        samples = pd.DataFrame(samples, columns=['Items'])

        distribution = samples.groupby(['Items']).size().reset_index(
            name='Frequency'
        )
        distribution = distribution.sort_index(ascending=False)
        distribution['Probability'] = (
                100*distribution.Frequency.cumsum()
            ) / distribution.Frequency.sum()

        mc_results = {}
        # Get nearest neighbor result
        for percentil in cfg['Montecarlo']['Percentiles'].get():
            result_index = distribution['Probability'].sub(percentil).abs()\
                .idxmin()
            mc_results[percentil] = distribution.loc[result_index, 'Items']

        return mc_results
    else:
        return None


def calc_simul_days():
    start = cfg['Montecarlo']['Simulation Start Date'].get()
    end = cfg['Montecarlo']['Simulation End Date'].get()
    return (end - start).days


def gather_metrics_data(jql_query):
    jira = atlassian_auth()
    issue = jql_search(jira, jql_query)
    dictio = convert_cfd_table(issue)
    kanban_data = read_dates(dictio)

    return kanban_data


def metrics():
    jql_query = str(cfg['Query'])
    simulations = cfg['Montecarlo']['Simulations'].get()
    mc_sources = cfg['Montecarlo']['Source'].get()
    kanban_data = gather_metrics_data(jql_query)
    ct = calc_cycletime_percentile(kanban_data, 85)
    tp = calc_throughput(kanban_data)
    mc = simulate_montecarlo(
        tp, sources=mc_sources,
        simul=simulations,
        simul_days=14)
    """
    if tp is not None:
        tp = tp.sum(axis=0)
    """
    return mc


def get_dict_value(dict, key1, key2, default=None):
    if not dict or dict[key1] is None or dict[key1][key2] is None:
        return default
    else:
        return dict[key1][key2]


def main():
    if os.path.exists("config"):
        for root, dirs, files in os.walk("config"):
            for name in files:
                cfg.set_file(os.path.join(root, name))
                print("Processing: {}".format(os.path.join(root, name)))
                mc = metrics()
                print(mc)
    elif os.path.isfile('config_test.yml'):
        cfg.set_file('config_test.yml')
    else:
        raise Exception("You don't have any valid config files")


if __name__ == "__main__":
    main()
