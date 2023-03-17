import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# streamlit run main.py

def main():
    st.set_page_config(page_title="Students Marks Prediction",
                       page_icon=":bar_chart:",
                       layout="wide")

    df = pd.read_excel(io='student-mat.xlsx',
                       engine='openpyxl',
                       sheet_name=0,
                       index_col=False,
                       keep_default_na=True
                       )

    st.sidebar.header("Please Filter Here:")
    sex = st.sidebar.multiselect(
        "Select The Sex:",
        options=df["sex"].unique(),
        default=df["sex"].unique()
    )

    famsize = st.sidebar.multiselect(
        "Select The Family Size",
        options=df["famsize"].unique(),
        default=df["famsize"].unique()
    )

    pstatus = st.sidebar.multiselect(
        "Select Parental Status",
        options=df["Pstatus"].unique(),
        default=df["Pstatus"].unique()
    )

    mjob = st.sidebar.multiselect(
        "Select Mother's Job",
        options=df["Mjob"].unique(),
        default=df["Mjob"].unique()
    )

    fjob = st.sidebar.multiselect(
        "Select Father's Job",
        options=df["Fjob"].unique(),
        default=df["Fjob"].unique()
    )

    xvalues = st.sidebar.multiselect(
        "Select Values For X",
        options=['G1', 'G2', 'G3','studytime', 'traveltime', 'absences', 'Dalc', 'health', 'failures', 'age', 'Medu', 'Fedu', 'famrel', 'freetime', 'goout', 'Walc'],
        default=['studytime', 'traveltime', 'absences', 'Dalc', 'health', 'failures']
    )

    df_selection = df.query(
        "sex == @sex & famsize == @famsize & Pstatus == @pstatus & Mjob == @mjob & Fjob == @fjob"
    )

    st.title(f"Columns Not Present In The DataSet Total {len(df_selection.columns)}")
    st.write(df_selection.columns)
    st.title("Complete View Of Selected DataSet")
    st.dataframe(df_selection)

    st.title("Charts Dashboard")
    st.markdown("##")

    total_study_time = int(df_selection["studytime"].sum())
    average_absence = round(df_selection["absences"].mean(), 1)
    total_g1_marks = int(df_selection["G1"].sum())
    total_g2_marks = int(df_selection["G2"].sum())
    total_g3_marks = int(df_selection["G3"].sum())
    total_daily_alco = int(df_selection["Dalc"].sum())
    total_weekly_alco = int(df_selection["Walc"].sum())

    clmn1, clmn2, clmn3, clmn4, clmn5, clmn6, clmn7 = st.columns(7)

    with clmn1:
        st.subheader("StudyTime:")
        st.subheader(f"Total {total_study_time}")

    with clmn2:
        st.subheader("Absences:")
        st.subheader(f"Avg {average_absence}")

    with clmn3:
        st.subheader("G1 Marks:")
        st.subheader(f"Total {total_g1_marks}")

    with clmn4:
        st.subheader("G2 Marks:")
        st.subheader(f"Total {total_g2_marks}")

    with clmn5:
        st.subheader("G3 Marks:")
        st.subheader(f"Total {total_g3_marks}")

    with clmn6:
        st.subheader("D-Alc-Con:")
        st.subheader(f"Total {total_daily_alco}")

    with clmn7:
        st.subheader("W-Alc-Con:")
        st.subheader(f"Total {total_weekly_alco}")

    st.markdown("---")

    chart = px.bar(
        df_selection,
        x="studytime",
        y="G1",
        orientation="v",
        title="<b>G1 By StudyTime</b>",
        color_discrete_sequence=["#0083B8"] * len(df_selection['G1']),
        template="plotly_white"
    )
    st.title("Bar Chart")
    st.plotly_chart(chart)

    fig = px.scatter(df_selection, x="studytime", y="G1", size="G1", color="guardian", hover_name="guardian",
                     log_x=True,
                     size_max=20, range_x=[1, 5], range_y=[1, 20])
    st.title("Scatter Plot")
    st.write(fig)

    fig2 = px.line(data_frame=df_selection, x="studytime", y="G1")
    st.title("Line Graph")
    st.plotly_chart(fig2, use_container_width=True)

    st.title("Heatmap Studytime | Failures")
    st.write(px.density_heatmap(df_selection, x="studytime", y="failures"))
    st.title("Heatmap Freetime | Failures")
    st.write(px.density_heatmap(df_selection, x="freetime", y="failures"))
    st.title("Heatmap Absences | Failures")
    st.write(px.density_heatmap(df_selection, x="absences", y="failures"))
    # st.write(px.density_heatmap(df_selection[['age', 'Medu', 'Fedu', 'traveltime','studytime', 'failures','famrel', 'freetime', 'goout', 'Dalc', 'Walc','health', 'absences', 'G1', 'G2', 'G3']],))

    st.title("Pairplot")
    st.pyplot(sns.pairplot(data=df_selection,
                           x_vars=["studytime", "freetime", "health", "age", "Dalc", "Walc", "absences", "G1", "G2",
                                   "G3"], y_vars=["G1", "G2", "G3"]))

    st.title("Studytime Vs G1")
    lmpt = sns.lmplot(x='studytime', y='G1', data=df_selection)
    st.pyplot(lmpt)
    st.title("Studytime Vs G2")
    lmpt1 = sns.lmplot(x='studytime', y='G2', data=df_selection)
    st.pyplot(lmpt1)
    st.title("Studytime Vs G3")
    lmpt2 = sns.lmplot(x='studytime', y='G3', data=df_selection)
    st.pyplot(lmpt2)
    st.title("G1 Vs G3")
    lmpt3 = sns.lmplot(x='G1', y='G3', data=df_selection)
    st.pyplot(lmpt3)
    st.title("G1 Vs G2")
    lmpt3 = sns.lmplot(x='G1', y='G2', data=df_selection)
    st.pyplot(lmpt3)

    # Model Train Test and Linear Regression Below

    #x = df_selection[['studytime', 'traveltime', 'absences', 'Dalc', 'health', 'failures'] + xvalues]
    x = df_selection[xvalues]
    y1 = df_selection['G1']
    y2 = df_selection['G2']
    y3 = df_selection['G3']

    lg = LinearRegression()
    rf = RandomForestRegressor(n_estimators=500, bootstrap=True, max_depth=50, max_features=4, min_samples_leaf=7,
                               min_samples_split=10)
    x_train, x_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.80, random_state=0)

    lg.fit(x_train, y1_train)
    rf.fit(x_train, y1_train)

    st.title('Coefficient Values')
    st.write(lg.coef_)

    predictions1 = lg.predict(x_test)
    rf_predictions1 = rf.predict(x_test)
    # testing our prediction against y1(G1)
    st.title("Scatter Plot Of Y1 Test values VS Predicted Values | LinearRegression Model")
    st.title("X = Y1 Test, Y=Prediction1")
    fig1 = px.scatter(x=y1_test, y=predictions1)
    st.plotly_chart(fig1)

    st.title("Scatter Plot Of Y1 Test values VS Predicted Values | RandomForestRegressor")
    st.title("X = Y1 Test, Y=RF_Prediction1")
    rf_fig1 = px.scatter(x=y1_test, y=rf_predictions1)
    st.plotly_chart(rf_fig1)

    # testing the same model over y2(G2)
    x_train, x_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.80, random_state=0)
    predictions2 = lg.predict(x_test)
    rf_predictions2 = rf.predict(x_test)
    st.title("Scatter Plot Of Y2 Test values VS Predicted Values | LinearRegression Model")
    st.title("X = Y2 Test, Y=Prediction2")
    fig2 = px.scatter(x=y2_test, y=predictions2)
    st.plotly_chart(fig2)

    st.title("Scatter Plot Of Y1 Test values VS Predicted Values | RandomForestRegressor")
    st.title("X = Y2 Test, Y=RF_Prediction2")
    rf_fig2 = px.scatter(x=y1_test, y=rf_predictions2)
    st.plotly_chart(rf_fig2)

    # For predicting over y3(G3) it is mentioned to train first over y1 and y2.
    lg.fit(x_train, y2_train)
    x_train, x_test, y3_train, y3_test = train_test_split(x, y3, train_size=0.80, random_state=0)
    predictions3 = lg.predict(x_test)
    rf_predictions3 = rf.predict(x_test)
    st.title("Scatter Plot Of Y1 Test values VS Predicted Values | LinearRegression Model")
    st.title("X = Y3 Test, Y=Prediction3")
    fig3 = px.scatter(x=y3_test, y=predictions3)
    st.plotly_chart(fig3)

    st.title("Scatter Plot Of Y1 Test values VS Predicted Values | RandomForestRegressor")
    st.title("X = Y3 Test, Y=RF_Prediction3")
    rf_fig3 = px.scatter(x=y3_test, y=rf_predictions3)
    st.plotly_chart(rf_fig3)

    # Let see how much error we were having

    st.title(f"RMSE Y1 Test, Prediction1:      {np.sqrt(metrics.mean_squared_error(y1_test, predictions1))}")
    st.title(f"RMSE Y2 Test, Prediction2:      {np.sqrt(metrics.mean_squared_error(y2_test, predictions2))}")
    st.title(f"RMSE Y3 Test, Prediction3:      {np.sqrt(metrics.mean_squared_error(y3_test, predictions3))}")

    st.title(f'R2 Score For Y1_Test:      {r2_score(y1_test, (predictions1 * 0.3 + predictions2 * 0.3 + predictions3 * 0.3))}')
    st.title(f'R2 Score For Y2_Test:      {r2_score(y2_test, (predictions1 * 0.3 + predictions2 * 0.3 + predictions3 * 0.3))}')
    st.title(f'R2 Score For Y3_Test:      {r2_score(y3_test, (predictions1 * 0.3 + predictions2 * 0.3 + predictions3 * 0.3))}')


if __name__ == '__main__':
    main()
