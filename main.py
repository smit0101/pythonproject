import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression








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

    df_selection = df.query(
        "sex == @sex & famsize == @famsize & Pstatus == @pstatus & Mjob == @mjob & Fjob == @fjob"
    )

    st.write(df_selection.columns)
    st.dataframe(df_selection)


    st.title("Charts Dashboard")
    st.markdown("##")

    total_study_time= int(df_selection["studytime"].sum())
    average_absence = round(df_selection["absences"].mean(),1)
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
        x = "studytime",
        y= "G1",
        orientation="v",
        title="<b>G1 By StudyTime</b>",
        color_discrete_sequence=["#0083B8"]*len(df_selection),
        template="plotly_white"
    )
    st.title("Bar Chart")
    st.plotly_chart(chart)

    fig = px.scatter(df_selection, x="studytime", y="G1", size="G1", color="guardian", hover_name="guardian", log_x=True,
                     size_max=20, range_x=[1,5], range_y=[1,20])
    st.title("Scatter Plot")
    st.write(fig)

    fig2 = px.line(data_frame=df_selection,x="studytime",y="G1")
    st.title("Line Graph")
    st.plotly_chart(fig2, use_container_width=True)

    st.title("Heatmap Studytime | Failures")
    st.write(px.density_heatmap(df_selection,x="studytime",y="failures"))
    st.title("Heatmap Freetime | Failures")
    st.write(px.density_heatmap(df_selection, x="freetime", y="failures"))
    st.title("Heatmap Romantic | Studytime")
    st.write(px.density_heatmap(df_selection, x="romantic", y="studytime"))
    st.title("Pairplot")
    st.pyplot(sns.pairplot(data=df_selection,x_vars=["studytime","freetime","health", "age","Dalc", "Walc"],y_vars=["G1","G2","G3"]))
    st.title("Studytime Vs FreeTime")
    lmpt = sns.lmplot(x='Walc', y='Dalc', data=df_selection)
    st.pyplot(lmpt)

    y = df_selection['G1']
    x = df_selection[['studytime', 'freetime', 'absences', 'Dalc', 'health', 'failures' ]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    lg = LinearRegression()
    lg.fit(x_train,y_train)
    st.write(lg.coef_)
    predictions = lg.predict(x_test)
    ans = plt.scatter(y_test,predictions)
    plt.xlabel('y test')
    plt.ylabel('Predicted')
    st.pyplot(ans)

if __name__ == '__main__':
    main()



