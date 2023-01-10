import os
import sys
from itertools import combinations

import numpy as np
import pandas
import plotly.graph_objects as go
import plotly.io as io
import sqlalchemy
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from sklearn import linear_model, metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# noinspection PyPackageRequirements
def main():

    print("start pythonfile")

    path = os.path.dirname(os.path.realpath(__file__))

    db_user = "root"
    db_pass = "ma"  # pragma: allowlist secret
    db_host = "mydb"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
        SELECT *
        from result

    """
    df = pandas.read_sql_query(query, sql_engine)

    df = df.fillna(df.mean())
    # tempist = df["temp"].values.tolist()
    # templist = [s.strip('degrees') for s in tempist]
    # df["temp"] = templist

    # df['plate_appear'] = df['plate_appear'].astype(float)
    df["diff_plate"] = df["a_plate"] - df["h_plate"]
    df["diff_a2b-h1b"] = df["a_2b"] - df["h_1b"]
    df["diff_finalscore"] = df["h_finalscore"] - df["a_finalscore"]
    df["diff_slugging_avg"] = df["h_slugging_avg"] - df["a_slugging_avg"]
    df["diff_axbh-htob"] = df["a_xbh"] - df["h_tob"]
    # df["diff_atb-htb"] = df["a_tb"] - df["h_tb"]
    df["diff_bat_avg"] = df["h_batting_avg"] - df["a_batting_avg"]
    df["diff_hbase-axbh"] = df["h_baserun"] - df["a_xbh"]
    df["diff_baserun"] = df["a_baserun"] - df["h_baserun"]

    df["diff_walk2strikeout_ratio"] = (
        df["h_walk2strikeout_ratio"] - df["a_walk2strikeout_ratio"]
    )
    df["diff_groundfly_ratio"] = df["a_groundfly_ratio"] - df["h_groundfly_ratio"]
    df["diff_strike_to_walk"] = df["a_strike_to_walk"] - df["h_p_strike_to_walk"]
    df["diff_dpip-hpip"] = df["a_p_ip"] - df["h_p_ip"]
    df["diff_hpip-aiso"] = df["h_p_ip"] - df["a_iso"]
    df["diff_aiso-apip"] = df["a_iso"] - df["a_p_ip"]
    # df["diff_slugging_per"] = df["h_slugging_percentage"] - df["a_slugging_percentage"]
    df["diff_hstike2walk-hiso"] = df["h_strike_to_walk"] - df["h_iso"]
    # df["diff_hslug-abase"] = df["h_slugging_avg"] - df["a_baserun"]
    df["diff_hiso-abase"] = df["h_iso"] - df["a_baserun"]
    df["diff_hiso-abase"] = df["a_strike_to_walk"] - df["a_baserun"]

    df.drop("gameid", inplace=True, axis=1)
    df.drop("date", inplace=True, axis=1)
    df.drop("h_team", inplace=True, axis=1)
    df.drop("a_team", inplace=True, axis=1)
    df.drop("home_streak", inplace=True, axis=1)
    df.drop("away_streak", inplace=True, axis=1)
    df.drop("a_ab", inplace=True, axis=1)
    df.drop("h_ab", inplace=True, axis=1)
    df.drop("h_3b", inplace=True, axis=1)
    df.drop("a_plate", inplace=True, axis=1)
    df.drop("a_tb", inplace=True, axis=1)
    df.drop("h_tb", inplace=True, axis=1)
    df.drop("a_xbh", inplace=True, axis=1)
    df.drop("a_1b", inplace=True, axis=1)
    df.drop("a_2b", inplace=True, axis=1)
    df.drop("a_3b", inplace=True, axis=1)
    df.drop("a_sf", inplace=True, axis=1)
    df.drop("h_1b", inplace=True, axis=1)
    df.drop("h_2b", inplace=True, axis=1)
    df.drop("h_xbh", inplace=True, axis=1)
    df.drop("h_sf", inplace=True, axis=1)
    df.drop("a_bb", inplace=True, axis=1)
    df.drop("h_bb", inplace=True, axis=1)
    df.drop("a_tob", inplace=True, axis=1)
    df.drop("temp", inplace=True, axis=1)
    df.drop("h_babip", inplace=True, axis=1)
    df.drop("h_plate", inplace=True, axis=1)
    df.drop("h_strikeout", inplace=True, axis=1)
    df.drop("h_tob", inplace=True, axis=1)
    df.drop("a_strikeout", inplace=True, axis=1)
    df.drop("diff_xbh", inplace=True, axis=1)
    df.drop("a_walk2strikeout_ratio", inplace=True, axis=1)
    df.drop("h_walk2strikeout_ratio", inplace=True, axis=1)
    df.drop("h_groundfly_ratio", inplace=True, axis=1)
    df.drop("a_slugging_avg", inplace=True, axis=1)
    df.drop("a_finalscore", inplace=True, axis=1)
    df.drop("away_pitch_substute", inplace=True, axis=1)
    df.drop("home_pitch_substute", inplace=True, axis=1)
    df.drop("h_p_ip", inplace=True, axis=1)
    df.drop("a_p_ip", inplace=True, axis=1)

    df = df.astype(float)

    response = "hometeam_win"

    # print(list(df.columns))
    X = df.loc[:, df.columns != response]
    y = df[response]

    finaltable = pandas.DataFrame(
        index=list(X.columns),
        columns=["feature_names", "mse", "wmse", "t_value", "p_value", "random_forest"],
    )
    finaltable["feature_names"] = list(X.columns)

    # con/con corr
    corr_p = X.corr("pearson")
    dff1 = corr_p.stack().reset_index(name="value")
    # print(dff1)
    # print(corr_p)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=corr_p.columns, y=corr_p.index, z=np.array(corr_p)))
    # fig.show()
    con_con_cor_plot = io.to_html(fig, include_plotlyjs="cdn")
    dff1 = dff1.sort_values(by="value", ascending=False)

    # brute force con/con

    final2 = []

    for i1, i2 in combinations(X.columns, 2):
        # ddf = pandas.cut(con_X.i1, bins=2)
        # print([i1, i2])
        ddf1 = pandas.cut(X[i1], bins=10)
        ddf2 = pandas.cut(X[i2], bins=10)
        ldf2 = pandas.DataFrame()
        ldf2["X1"] = ddf1
        ldf2["X2"] = ddf2
        ldf2["response"] = y
        # print(ldf2)
        thislist2 = []
        count_l2 = []
        for key, group in ldf2.groupby(["X1", "X2"]):
            # print(i1, i2)
            # print(key)
            # print(group)
            calcc = np.array(group["response"])
            mse = calcc.mean()
            count_l2.append(len(group))
            thislist2.append([key[0], key[1], mse])
            thatlist2 = pandas.DataFrame(thislist2, columns=["X1", "X2", "mse"])
            thoselist2 = thatlist2.pivot(index="X1", columns="X2", values="mse")
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=thoselist2.columns.astype("str"),
                y=thoselist2.index.astype("str"),
                z=np.array(thoselist2),
            )
        )
        # fig.show()
        fig.write_html(f"boo{i1}{i2}.html")
        a2 = np.array(thatlist2["mse"])
        mean_m2 = a2 - np.mean(y)
        mean_mm2 = np.square(mean_m2)
        mean_m2 = sum(mean_mm2)
        mean_m2 = mean_m2 / len(a2)

        mean_wm2 = np.multiply(mean_mm2, np.array(count_l2) / sum(count_l2))
        mean_wm2 = sum(mean_wm2)

        final2.append(
            [i1, i2, f"<a href='//{path}/boo{i1}{i2}.html'>{mean_m2}</a>", mean_wm2]
        )
    finaldf2 = pandas.DataFrame(
        final2, columns=["Predictor1", "Predictor2", "mse", "wmse"]
    )
    finaldf2 = finaldf2.sort_values(by="wmse", ascending=False)

    plottable = pandas.DataFrame()
    plotslist = []

    # plot con/cat
    plot_df = X
    plot_df["target"] = y

    plot_dfoc = plot_df.loc[:, ~plot_df.T.duplicated(keep="last")]

    for i in plot_dfoc.iloc[:, :-1]:
        hist_data = []
        classes = plot_dfoc.groupby("target")[i].apply(list)
        for a in classes:
            hist_data.append(a)

        group_labels = list(plot_dfoc.groupby("target").groups.keys())

        # Create distribution plot with custom bin_size
        fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
        fig_1.update_layout(
            title="Continuous Predictor by Categorical Response",
            xaxis_title="Predictor",
            yaxis_title="Distribution",
        )
        # fig_1.show()
        fig_1.write_html(f"hw4oc{i}.html")
        plotslist.append(f"<a href='//{path}/hw4oc{i}.html'>{i}</a>")

    plottable["predictors"] = plotslist

    X.drop("target", inplace=True, axis=1)

    # mean of response mse

    for att in X.columns:
        pindf = pandas.DataFrame()
        # print(df[att])
        pins = pandas.cut(df[att], bins=10)
        pindf["X1"] = pins
        pindf["response"] = y
        alisted = []
        # print(pins)
        for key, group in pindf.groupby(["X1"]):
            # print(group)
            caly = np.array(group["response"])
            mee = np.mean(caly)
            alisted.append([key, mee, len(group)])
            blisted = pandas.DataFrame(alisted, columns=["key", "diff mean", "pop"])
            blisted["bin_center"] = blisted["key"].apply(lambda x: x.mid)
            blisted["popmean"] = np.mean(y)

        fig_2 = go.Figure(
            layout=go.Layout(
                title="Binned difference with mean of response vs mean",
                yaxis2=dict(overlaying="y"),
            )
        )
        fig_2.add_trace(go.Bar(x=blisted["bin_center"], y=blisted["pop"], yaxis="y1"))
        fig_2.add_trace(
            go.Scatter(
                x=blisted["bin_center"],
                y=blisted["diff mean"],
                yaxis="y2",
                mode="lines",
                line=go.scatter.Line(color="red"),
            )
        )
        fig_2.add_trace(
            go.Scatter(
                x=blisted["bin_center"],
                y=blisted["popmean"],
                yaxis="y2",
                mode="lines",
                line=go.scatter.Line(color="green"),
                showlegend=False,
            )
        )
        fig_2.write_html(f"mor{att}.html")

        ac = np.array(blisted["diff mean"])
        calcu = ac - np.mean(y)
        calcuu = np.square(calcu)
        calcu = np.nansum(calcuu)
        calcu = calcu / len(ac)

        w_calcu = np.multiply(
            calcuu, (np.array(blisted["pop"]) / np.array(blisted["pop"]).sum())
        )
        w_calcu = np.nansum(w_calcu)

        finaltable.loc[att, "mse"] = f"<a href='//{path}/mor{att}.html'>{calcu}</a>"
        finaltable.loc[att, "wmse"] = w_calcu

    stat_X = np.array(X)
    for idx, column in enumerate(stat_X.T):
        feature_name = list(X.columns)[idx]
        predictor = statsmodels.api.add_constant(column)
        linear_regression_model = statsmodels.api.Logit(y, predictor, missing="drop")
        linear_regression_model_fitted = linear_regression_model.fit()
        # print(f"Variable: {feature_name}")
        # print(linear_regression_model_fitted.summary())

        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        # print(t_value)
        # print(p_value)
        finaltable.loc[
            feature_name, "t_value"
        ] = f"<a href='//{path}/stat{feature_name}.html'>{t_value}</a>"
        finaltable.loc[
            feature_name, "p_value"
        ] = f"<a href='//{path}/stat{feature_name}.html'>{p_value}</a>"

        # Plot the figure
        fig_6 = px.scatter(x=column, y=y, trendline="lowess")
        fig_6.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        # fig_6.show()
        fig_6.write_html(f"stat{feature_name}.html")

    rf = RandomForestRegressor(n_estimators=50)
    rf.fit(X, y)

    for i in range(len(list(X.columns))):
        finaltable.loc[list(X.columns)[i], "random_forest"] = rf.feature_importances_[i]

    fig_7 = px.bar(x=list(X.columns), y=rf.feature_importances_)
    fig_7.update_layout(
        title="forest ranking",
        xaxis_title="predictor",
        yaxis_title="y",
    )
    rf_plot = io.to_html(fig_7, include_plotlyjs="cdn")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, shuffle=False
    )

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
    f1_score(y_test, y_pred)

    logr = linear_model.LogisticRegression()
    logr.fit(X_train, y_train)
    predicted = logr.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, predicted))
    f1_score(y_test, predicted)

    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(X_train, y_train)
    y_pred_5 = knn5.predict(X_test)
    print("Accuracy with k=5", metrics.accuracy_score(y_test, y_pred_5))
    print(f1_score(y_test, y_pred_5))

    clf = SVC(kernel="linear")
    # fitting x samples and y classes
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    print("svc", metrics.accuracy_score(y_test, clf_pred))
    print(f1_score(y_test, clf_pred))

    model = GaussianNB()
    model.fit(X_train, y_train)
    g_predict = model.predict(X_test)
    print("Naive Bayes Classification", metrics.accuracy_score(y_test, g_predict))
    print(f1_score(y_test, g_predict))

    dp = DecisionTreeClassifier()
    dp = dp.fit(X_train, y_train)
    de_pred = dp.predict(X_test)
    print("decision tree", metrics.accuracy_score(y_test, de_pred))
    print(f1_score(y_test, de_pred))

    with open("/share/output.html", "w") as _file:
        _file.write("<h1>Homework 5 Brian</h1>")

        _file.write("<h1>con/con correlation</h1>" + dff1.to_html())
        _file.write("<h1>con/con correlation plot</h1>" + con_con_cor_plot)

        _file.write(
            "<h1>brute force con/con</h1>"
            + finaldf2.to_html(index=False, render_links=True, escape=False)
        )

        _file.write(
            "<h1>HW4 plot table (all predictors)</h1>"
            + plottable.to_html(index=False, render_links=True, escape=False)
        )

        _file.write(
            "<h1>homework 4 table</h1>"
            + finaltable.to_html(index=False, render_links=True, escape=False)
        )

        _file.write("<h1>random forest plot</h1>" + rf_plot)

        _file.write(
            f"<h2>random forest model: accuracy{metrics.accuracy_score(y_test, y_pred)}</h2>"
            f"<h2>random forest model: f1 score{f1_score(y_test, y_pred)}</h2>"
        )
        _file.write(
            f"<h2>logistic regression model: accuracy{metrics.accuracy_score(y_test, predicted)}</h2>"
            f"<h2>logistic regression model: f1 score{f1_score(y_test, predicted)}</h2>"
        )

        _file.write(
            f"<h2>KNN model: accuracy{metrics.accuracy_score(y_test, y_pred_5)}</h2>"
            f"<h2>KNN model: f1 score{f1_score(y_test, y_pred_5)}</h2>"
        )
        _file.write(
            f"<h2>SVC model: accuracy{metrics.accuracy_score(y_test, clf_pred)}</h2>"
            f"<h2>SVC model: f1 score{f1_score(y_test, clf_pred)}</h2>"
        )
        _file.write(
            f"<h2>Naive Bayes model: accuracy{metrics.accuracy_score(y_test, g_predict)}</h2>"
            f"<h2>Naive Bayes model: f1 score{f1_score(y_test, g_predict)}</h2>"
        )
        _file.write(
            f"<h2>decision tree model: accuracy{metrics.accuracy_score(y_test, de_pred)}</h2>"
            f"<h2>decision tree model: f1 score{f1_score(y_test, de_pred)}</h2>"
        )


if __name__ == "__main__":
    sys.exit(main())
