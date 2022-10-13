import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def main():

    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    B = data["target"]

    B_df = pd.DataFrame(B)

    for att in df.columns:
        if df[att].dtype.kind in "OM":
            label = LabelEncoder()
            df[att] = label.fit_transform(df[att])

    if B.dtype.kind in "OM":
        label = LabelEncoder()
        B = label.fit_transform(B)

    y = np.array(B)

    cat_p = pd.DataFrame()
    con_p = pd.DataFrame()

    if (1.0 * B_df.nunique() / B_df.count() < 0.05)[0]:
        response = 1  # cat
    else:
        response = 0

    for att in df.columns:
        likely_cat = 1.0 * df[att].nunique() / df[att].count() < 0.05
        if not likely_cat:
            con_p[att] = df[att]
            print(att + " is con")
        else:
            cat_p[att] = df[att]
            print(att + " is cat")

    X_con = np.array(con_p)
    X_cat = np.array(cat_p)

    if not con_p.empty:
        print("there is con_p")

    if not cat_p.empty:
        print("there is cat_p")

    # plot con/cat
    if not con_p.empty and response == 1:
        plot_df = pd.DataFrame(con_p)
        plot_df["target"] = data["target"]

        for i in plot_df.iloc[:, :-1]:
            hist_data = []
            classes = plot_df.groupby("target")[i].apply(list)
            for i in classes:
                hist_data.append(i)

            group_labels = list(plot_df.groupby("target").groups.keys())

            # Create distribution plot with custom bin_size
            fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
            fig_1.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Predictor",
                yaxis_title="Distribution",
            )
            fig_1.show()

            n = len(plot_df)
            fig_2 = go.Figure()
            for curr_hist, curr_group in zip(hist_data, group_labels):
                fig_2.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group, n),
                        y=curr_hist,
                        name=curr_group,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
            fig_2.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Response",
                yaxis_title="Predictor",
            )
            # fig_2.show()

    # plot con/con
    if not con_p.empty and response == 0:
        for i in range(np.shape(X_con)[1]):
            fig = px.scatter(x=X_con[:, i], y=y, trendline="ols")
            fig.update_layout(
                title="Continuous Response by Continuous Predictor",
                xaxis_title="Predictor",
                yaxis_title="Response",
            )
            fig.show()

    # plot cat/con
    if not cat_p.empty and response == 0:
        plot_df = pd.DataFrame(con_p)
        plot_df["target"] = data["target"]
        for i in plot_df.iloc[:, :-1]:
            hist_data = []
            classes = plot_df.groupby("target")[i].apply(list)
            for i in classes:
                hist_data.append(i)

            group_labels = list(plot_df.groupby("target").groups.keys())
            fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
            fig_1.update_layout(
                title="Continuous Response by Categorical Predictor",
                xaxis_title="Response",
                yaxis_title="Distribution",
            )
            fig_1.show()

            n = len(plot_df)
            fig_2 = go.Figure()
            for curr_hist, curr_group in zip(hist_data, group_labels):
                fig_2.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group, n),
                        y=curr_hist,
                        name=curr_group,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
            fig_2.update_layout(
                title="Continuous Response by Categorical Predictor",
                xaxis_title="Groupings",
                yaxis_title="Response",
            )
            fig_2.show()

    # plot cat/cat
    if not cat_p.empty and response == 1:
        conf_matrix = confusion_matrix(X_cat[:, 1], y)

        fig_no_relationship = go.Figure(
            data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
        )
        fig_no_relationship.update_layout(
            title="Categorical Predictor by Categorical Response (with relationship)",
            xaxis_title="Response",
            yaxis_title="Predictor",
        )
        fig_no_relationship.show()

    # Regression: boolean #dataset:load_breast_cancer()

    if not con_p.empty and response == 1:
        for idx, column in enumerate(X_con.T):
            feature_name = list(con_p.columns)[idx]
            predictor = statsmodels.api.add_constant(column)
            linear_regression_model = statsmodels.api.Logit(y, predictor)
            linear_regression_model_fitted = linear_regression_model.fit()
            print(f"Variable: {feature_name}")
            print(linear_regression_model_fitted.summary())

            # Get the stats
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

            # Plot the figure
            fig = px.scatter(x=column, y=y, trendline="lowess")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title="y",
            )
            fig.show()

    # Regression: Continuous response
    if not con_p.empty and response == 0:
        for idx, column in enumerate(con_p.T):
            feature_name = list(con_p.columns)[idx]
            predictor = statsmodels.api.add_constant(column)
            linear_regression_model = statsmodels.api.OLS(y, predictor)
            linear_regression_model_fitted = linear_regression_model.fit()
            print(f"Variable: {feature_name}")
            print(linear_regression_model_fitted.summary())

            # Get the stats
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

            # Plot the figure
            fig = px.scatter(x=column, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title="y",
            )
            fig.show()

    # random forest ranking
    if not con_p.empty:
        rf = RandomForestRegressor(n_estimators=50)
        rf.fit(X_con, y)
        print(rf.feature_importances_)
        plt.barh(list(con_p.columns), rf.feature_importances_)
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
