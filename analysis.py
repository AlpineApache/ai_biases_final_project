import pandas as pd
from scipy.stats import ttest_ind

race_result_list = []
race_mean_differences = []
libraries = ["vader_sentiment", "text_blob"]
emotions = ["anger", "joy", "fear", "sadness"]
genders = ["female", "male"]
races = ["European", "African-American"]

# documentation https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
def run_t_test(category_1, category_2):
	temp_significance = []
	temp_hypothesis = []

	# significance value and degrees of freedom
	sig = 0.025  # (0.05/2 for two-sided test)
	# degrees of freedom = 1050
	# rejection Hypothesis area: +-1.98
	rej = 1.98

	# actual t-test
	result = ttest_ind(category_1, category_2, axis=0, equal_var=1, nan_policy="propagate")
	statistic = result[0]
	pvalue = result[1]

	# significance test
	if pvalue < sig:
		temp_significance.append(0)
		temp_significance.append("Significative")
	else:
		temp_significance.append(0)
		temp_significance.append("Not Significative")

	# hypothesis rejection
	if statistic > 0 and statistic > rej:
		temp_hypothesis.append(0)
		temp_hypothesis.append("Rejected")
	elif statistic < 0 and statistic < -rej:
		temp_hypothesis.append(0)
		temp_hypothesis.append("Rejected")
	else:
		temp_hypothesis.append(0)
		temp_hypothesis.append("Confirmed")

	return (
		temp_significance,
		temp_hypothesis
	)


def aggregate_racial_bias_results(data):
	#####
	# Calculate mean along the following parameters: library x emotion x race
	for library in libraries:
		for emotion in emotions:
			for race in races:
				gender_df = data.loc[
					(data["Library"] == library) & (data["Race"] == race) & (data["Emotion"] == emotion)]
				temp_data = {
					"Library": library,
					"Race": race,
					"Emotion": emotion,
					"Mean": gender_df["Compound"].mean()
				}
				race_result_list.append(temp_data)

	racial_bias_results = pd.DataFrame(race_result_list)

	#####
	# Calculate differences between European American and African American mean
	for i in racial_bias_results.Mean:
		if len(race_mean_differences) % 2 == 0:
			race_mean_differences.append(i)
		else:
			race_mean_differences.append(i - (race_mean_differences[-1]))

	difference = []
	for i in race_mean_differences:
		if len(difference) % 2 == 0:
			# differences=[]
			difference.append(0)
		else:
			difference.append(i)

	difference = pd.Series(difference)
	racial_bias_results = racial_bias_results.assign(Difference=difference)

	#####
	# Label difference between Male/female means
	prevalence = []
	for n in racial_bias_results.Difference:
		if not len(prevalence) % 2 == 0:
			if n == 0:
				prevalence.append("EA=AA")
			else:
				if n > 0:
					prevalence.append("EA<AA")
				if n < 0:
					prevalence.append("EA>AA")
		else:
			prevalence.append(0)

	prevalence = pd.Series(prevalence)
	racial_bias_results = racial_bias_results.assign(Prevalence=prevalence)

	#####
	# Run t-test on race bias results
	significance = []
	hypothesis = []
	for library in libraries:
		for emotion in emotions:
			african_var = data.loc[(data["Library"] == library) & (data["Race"] == "African-American") & (data["Emotion"] == emotion)]
			european_var = data.loc[(data["Library"] == library) & (data["Race"] == "European-American") & (data["Emotion"] == emotion)]
			category_1 = african_var["Compound"]
			category_2 = european_var["Compound"]
			temp = run_t_test(category_1, category_2)
			significance.extend(temp[0])
			hypothesis.extend(temp[1])

	temp_significance = pd.Series(significance)
	racial_bias_results = racial_bias_results.assign(Significativity=temp_significance)

	temp_hypothesis = pd.Series(hypothesis)
	racial_bias_results = racial_bias_results.assign(Hypothesis=temp_hypothesis)

	#####
	# Export race bias results to .csv file
	racial_bias_results.to_csv("racial_bias_results.csv")


def t_test_bias_analysis(data):
	aggregate_racial_bias_results(data)
