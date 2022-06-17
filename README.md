# Jira metrics
A simple python script that allows you to extract three main team metrics from Jira Cloud API and print them in a google slide:
* Throughput
* Cycle time
* Monte Carlo forecast

Before run the script you'll need:
* Jira API Token

## Jira API Token
* Go to the url https://id.atlassian.com/manage-profile/security/api-tokens
* Click on the button "Create API Token", add any label you wish to identify the application (Ex.: jira_metrics)
* Copy the token store in a safe location 
* Copy the token to each of your config files

## Running the script
You should run on Python 3.7, to check your python version:
```shell
python3 --version
```
Make sure you had the requirements.txt installed (Docker container is comming soon):
```shell
pip3 install -r requirements.txt
```
You can run the script with the command:
```shell
./app/jira_metrics.py
```