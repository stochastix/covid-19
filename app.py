import numpy as np
#https://github.com/matplotlib/matplotlib/issues/14304
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from scipy import stats as sps
from scipy.interpolate import interp1d
import pandas as pd
import json
from datetime import datetime
import urllib.request
from flask import render_template
from flask import Flask
import os

current_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)

state_names = {
 'an': 'Andaman & Nicobar',
 'ap': 'Andhra Pradesh',
 'ar': 'Arunachal Pradesh',
 'as': 'Assam',
 'br': 'Bihar',
 'ch': 'Chandigarh',
 'ct': 'Chhattisgarh',
 'dd': 'Diu & Daman',
 'dl': 'Delhi',
 'dn': 'Dadra & Nagar Haveli',
 'ga': 'Goa',
 'gj': 'Gujarat',
 'hp': 'Himachal Pradesh',
 'hr': 'Haryana',
 'tt': 'India',
 'jh': 'Jharkhand',
 'jk': 'Jammu & Kashmir',
 'ka': 'Karnataka',
 'kl': 'Kerala',
 'la': 'Ladakh',
 'ld': 'Lakshadweep',
 'mh': 'Maharashtra',
 'ml': 'Meghalaya',
 'mn': 'Manipur',
 'mp': 'Madhya Pradesh',
 'mz': 'Mizoram',
 'nl': 'Nagaland',
 'or': 'Orissa',
 'pb': 'Punjab',
 'py': 'Pondicherry',
 'rj': 'Rajasthan',
 'sk': 'Sikkim',
 'tg': 'Telangana',
 'tn': 'Tamil Nadu',
 'tr': 'Tripura',
 'up': 'Uttar Pradesh',
 'ut': 'Uttarakhand',
 'wb': 'West Bengal'
}


def highest_density_interval(pmf, p=.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if (isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]

    return pd.Series([low, high],
                     index=[f'Low_{p * 100:.0f}',
                            f'High_{p * 100:.0f}'])
def prepare_cases(cases, cutoff=10):
    new_cases = cases.diff()
    smoothed = new_cases.rolling(7,
                                 win_type='gaussian',
                                 min_periods=1,
                                 center=True).mean(std=2).round()
    idx_start = np.searchsorted(smoothed, cutoff)
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    return original, smoothed

def get_posteriors(sr, sigma=0.15):
    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:].values, lam),
        index=r_t_range,
        columns=sr.index[1:])

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                              ).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood

def plot_rt(result, ax, state_name):
    ax.set_title(f"{state_name}")

    # Colors
    ABOVE = [1, 0, 0]
    MIDDLE = [1, 1, 1]
    BELOW = [0, 0, 0]
    cmap = ListedColormap(np.r_[
                              np.linspace(BELOW, MIDDLE, 25),
                              np.linspace(MIDDLE, ABOVE, 25)
                          ])
    color_mapped = lambda y: np.clip(y, .5, 1.5) - .5

    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values

    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)

    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low_90'].values,
                     bounds_error=False,
                     fill_value='extrapolate')

    highfn = interp1d(date2num(index),
                      result['High_90'].values,
                      bounds_error=False,
                      fill_value='extrapolate')

    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
                             end=index[-1] + pd.Timedelta(days=1))

    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1] + pd.Timedelta(days=1))
    fig.set_facecolor('w')


def save_plot():
    global fig, r_t_range, GAMMA, highest_density_interval
    # # Column vector of k
    k = np.arange(0, 70)[:, None]
    # # Different values of Lambda
    lambdas = [10, 20, 30, 40]
    # # Evaluated the Probability Mass Function (remember: poisson is discrete)
    y = sps.poisson.pmf(k, lambdas)
    #
    fig, ax = plt.subplots()
    # ax.set(title='Poisson Distribution of Cases\n $p(k|\lambda)$')
    # plt.plot(k, y,
    #          marker='o',
    #          markersize=3,
    #          lw=0)
    # plt.legend(title="$\lambda$", labels=lambdas);
    k = 20
    lam = np.linspace(1, 45, 90)
    likelihood = pd.Series(data=sps.poisson.pmf(k, lam),
                           index=pd.Index(lam, name='$\lambda$'),
                           name='lambda')
    # likelihood.plot(title=r'Likelihood $P\left(k_t=20|\lambda\right)$');
    k = np.array([20, 40, 55, 90])
    # We create an array for every possible value of Rt
    R_T_MAX = 12
    r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)
    # Gamma is 1/serial interval
    # https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
    # https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
    GAMMA = 1 / 7
    # Map Rt into lambda so we can substitute it into the equation below
    # Note that we have N-1 lambdas because on the first day of an outbreak
    # you do not know what to expect.
    lam = k[:-1] * np.exp(GAMMA * (r_t_range[:, None] - 1))
    # Evaluate the likelihood on each day and normalize sum of each day to 1.0
    likelihood_r_t = sps.poisson.pmf(k[1:], lam)
    likelihood_r_t /= np.sum(likelihood_r_t, axis=0)
    # Plot it
    # ax = pd.DataFrame(
    #     data=likelihood_r_t,
    #     index=r_t_range
    # ).plot(
    #     title='Likelihood of $R_t$ given $k$',
    #     xlim=(0, 10)
    # )
    # ax.legend(labels=k[1:], title='New Cases')
    # ax.set_xlabel('$R_t$');
    posteriors = likelihood_r_t.cumprod(axis=1)
    posteriors = posteriors / np.sum(posteriors, axis=0)
    columns = pd.Index(range(1, posteriors.shape[1] + 1), name='Day')
    posteriors = pd.DataFrame(
        data=posteriors,
        index=r_t_range,
        columns=columns)
    # ax = posteriors.plot(
    #     title='Posterior $P(R_t|k)$',
    #     xlim=(0, 10)
    # )
    # ax.legend(title='Day')
    # ax.set_xlabel('$R_t$');
    most_likely_values = posteriors.idxmax(axis=0)
    hdi = highest_density_interval(posteriors)
    # ax = most_likely_values.plot(marker='o',
    #                              label='Most Likely',
    #                              title=f'$R_t$ by day',
    #                              c='k',
    #                              markersize=4)
    # ax.fill_between(hdi.index,
    #                 hdi['Low_90'],
    #                 hdi['High_90'],
    #                 color='k',
    #                 alpha=.1,
    #                 lw=0,
    #                 label='HDI')
    # ax.legend();
    url = 'https://api.covid19india.org/states_daily.json'
    req = urllib.request.Request(url)
    daily_data = json.loads(urllib.request.urlopen(req).read())['states_daily']
    states_data = []
    non_state_keys = ['date', 'status']
    for row in daily_data:
        this_status = row['status']
        # ignore the other two statuses: deceased and recovered
        if this_status == 'Confirmed':
            # this_date = datetime.strftime(datetime.strptime(row['date'], '%d-%b-%y'), '%Y-%m-%d')
            this_date = datetime.strptime(row['date'], '%d-%b-%y').date()
            for item in row.keys():
                if not item in non_state_keys:
                    states_data.append([
                        state_names[item],
                        this_date,
                        int(row[item]) if row[item] else 0
                    ])
    states_data = sorted(states_data,
                         key=lambda states_data: states_data[0] + states_data[1].strftime('%Y-%m-%d'))
    column_labels = ['state', 'date', 'cases']
    states = pd.DataFrame.from_records(states_data,
                                       columns=['state', 'date', 'cases'],
                                       index=['state', 'date'])
    states = states.squeeze()
    states = states.groupby(['state', 'date']).sum().groupby(level=0).cumsum()
    # if the state has fewer than 30 cases in all OR
    # there have been fewer than 10 non-zero days
    FILTERED_REGIONS = set([])
    STATES_1000_PLUS = set([])
    today = datetime.now().date()
    for state_name in state_names.values():
        cumulative = states.xs(state_name).tail(1).item()
        daily = states.xs(state_name).diff()
        if cumulative < 25:
            FILTERED_REGIONS.add(state_name)
            print('Removing ' + state_name + ' because there are fewer than 25 cumulative cases')
        elif len(daily[daily > 0]) < 15:
            FILTERED_REGIONS.add(state_name)
            print('Removing ' + state_name + ' because there are fewer than 15 days with non zero cases')
        else:
            cases_began = daily[daily > 0].keys()[0].date()
            days_since_case1 = (today - cases_began).days
            # z_since_case1 = len(daily[-days_since_case1:][daily==0])
            z_7d = len(daily[-7:][daily == 0])
            if z_7d > 2:
                FILTERED_REGIONS.add(state_name)
                print('Removing ' + state_name + ' because there are at least 3 days of zero cases in the last week')
        if cumulative > 1000:
            STATES_1000_PLUS.add(state_name)

    # state_name = 'India'
    cases = states.xs(state_name).rename(f"{state_name} cases")
    original, smoothed = prepare_cases(cases)
    # original.plot(title=f"{state_name} New Cases per Day",
    #               c='k',
    #               linestyle=':',
    #               alpha=.5,
    #               label='Actual',
    #               legend=True,
    #               figsize=(500 / 72, 300 / 72))
    #
    # ax = smoothed.plot(label='Smoothed',
    #                    legend=True)
    #
    # ax.get_figure().set_facecolor('w')
    # Note that we're fixing sigma to a value just for the example
    posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)
    MAX_DATE = posteriors.axes[1][-1].strftime('%Y-%m-%d')
    MAX_DATE_MINUS_1 = posteriors.axes[1][-2].strftime('%Y-%m-%d')
    MAX_DATE_MINUS_2 = posteriors.axes[1][-3].strftime('%Y-%m-%d')
    # ax = posteriors.plot(title=f'{state_name} - Daily Posterior for $R_t$',
    #                      legend=False,
    #                      lw=1,
    #                      c='k',
    #                      alpha=.3,
    #                      xlim=(0.4, 6))
    # ax.plot(posteriors.filter(like=MAX_DATE), c='r')
    # ax.plot(posteriors.filter(like=MAX_DATE_MINUS_1), c='g')
    # ax.plot(posteriors.filter(like=MAX_DATE_MINUS_2), c='b')
    # ax.set_xlabel('$R_t$');
    # Note that this takes a while to execute - it's not the most efficient algorithm
    hdis = highest_density_interval(posteriors, p=.9)
    most_likely = posteriors.idxmax().rename('ML')
    # Look into why you shift -1
    result = pd.concat([most_likely, hdis], axis=1)
    fig, ax = plt.subplots(figsize=(600 / 72, 400 / 72))
    plot_rt(result, ax, state_name)
    ax.set_title(f'Real-time $R_t$ for {state_name}')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    sigmas = np.linspace(1 / 20, 1, 20)
    targets = ~states.index.get_level_values('state').isin(FILTERED_REGIONS)
    states_to_process = states.loc[targets]
    results = {}
    for state_name, cases in states_to_process.groupby(level='state'):
        print(state_name)
        new, smoothed = prepare_cases(cases, cutoff=10)
        if len(smoothed) == 0:
            new, smoothed = prepare_cases(cases, cutoff=3)
        result = {}

        # Holds all posteriors with every given value of sigma
        result['posteriors'] = []

        # Holds the log likelihood across all k for each value of sigma
        result['log_likelihoods'] = []
        try:
            for sigma in sigmas:
                posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
                result['posteriors'].append(posteriors)
                result['log_likelihoods'].append(log_likelihood)
            # Store all results keyed off of state name
            results[state_name] = result
        except IndexError as e:
            print('Encountered an error while computing posteriors for ' + state_name)
            print(str(e))
        except ValueError as e:
            print('Encountered an error while computing posteriors for ' + state_name)
            print(str(e))
    # Each index of this array holds the total of the log likelihoods for
    # the corresponding index of the sigmas array.
    total_log_likelihoods = np.zeros_like(sigmas)
    # Loop through each state's results and add the log likelihoods to the running total.
    for state_name, result in results.items():
        # don't use all-India data while aggregating across states
        if not state_name == 'India':
            total_log_likelihoods += result['log_likelihoods']
    # Select the index with the largest log likelihood total
    max_likelihood_index = total_log_likelihoods.argmax()
    # Select the value that has the highest log likelihood
    sigma = sigmas[max_likelihood_index]
    # Plot it
    fig, ax = plt.subplots()
    ax.set_title(f"Maximum Likelihood value for $\sigma$ = {sigma:.2f}");
    ax.plot(sigmas, total_log_likelihoods)
    ax.axvline(sigma, color='k', linestyle=":")
    final_results = None
    for state_name, result in results.items():
        posteriors = result['posteriors'][max_likelihood_index]
        try:
            hdis_90 = highest_density_interval(posteriors, p=.9)
            hdis_50 = highest_density_interval(posteriors, p=.5)
            most_likely = posteriors.idxmax().rename('ML')
            result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
            if final_results is None:
                final_results = result
            else:
                final_results = pd.concat([final_results, result])
            # clear_output(wait=True)
        except ValueError as e:
            print('Error while processing ' + state_name)
            print(str(e))
    ncols = 4
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 3))
    for i, (state_name, result) in enumerate(final_results.groupby('state')):
        plot_rt(result, axes.flat[i], state_name)
    fig.tight_layout()
    fig.set_facecolor('w')
    #plt.plot()
    fig.savefig(current_path+'/static/images/plot.png')




@app.route("/", methods=["GET"])
def plotv_iew():
    return render_template('home.html', name='new_plot', url='/static/images/plot.png')

@app.route("/get_plot", methods=["GET"])
def get_plot():
    save_plot()

if __name__ == '__main__':
    app.run()
