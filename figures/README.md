# Data Visualization

## SHAP
Using SHAP, we are able to determine what columns have the biggest effect on our target variable of response rate. Inputing ``-v True`` as a flag yields visualizations saved in the figures folder.

Take the following example where character cutoff number was 70. There are a number of observations to record here:
* the number of messages sent in a given conversation was the most indicative of whether more participants from the group joined to respond
* the length of the message inversely correlated with the response rate
* later hours, but not too late were the most effective in getting a higher response rate
* having emoji in conversation starter messages seems to effectively intice responses
* there is a slightly higher response rate if the message was sent earlier in the week (e.g. Monday)

![Image](https://github.com/noelkonagai/specialization_project/blob/master/figures/shap_summary_vary_char_70.png)
