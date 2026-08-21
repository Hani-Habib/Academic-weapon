import pandas as pd


def prepare_municipality_data(ifor41, ifor32):
    gini = ifor41[ifor41["KOMMUNEDK"] != "Hele landet"].copy()
    decile = ifor32[ifor32["KOMMUNEDK"] != "Hele landet"].copy()

    gini["INDHOLD"] = pd.to_numeric(gini["INDHOLD"])
    decile["INDHOLD"] = pd.to_numeric(decile["INDHOLD"])

    decile = decile.pivot(
        index=["KOMMUNEDK", "TID"],
        columns="DECILGEN",
        values="INDHOLD"
    ).reset_index()

    deciles = [f"{i}. decil" for i in range(1, 11)]
    decile["Top 10 share"] = (
        decile["10. decil"] / decile[deciles].sum(axis=1) * 100
    )

    data = pd.merge(
        gini[["KOMMUNEDK", "TID", "INDHOLD"]],
        decile[["KOMMUNEDK", "TID", "Top 10 share"]],
        on=["KOMMUNEDK", "TID"]
    )

    data.columns = ["Municipality", "Year", "Gini", "Top 10 share"]
    data["Year"] = pd.to_numeric(data["Year"])

    return data


def prepare_education_data(latest, education):
    education = education.copy()
    education["INDHOLD"] = pd.to_numeric(education["INDHOLD"], errors="coerce")

    education = education[
        (education["HERKOMST"] == "I alt") &
        (education["ALDER"] == "Alder i alt") &
        (education["KØN"] == "I alt")
    ]

    total = education[
        education["HFUDD"] == "I alt"
    ][["BOPOMR", "INDHOLD"]]
    total.columns = ["Municipality", "Total"]

    high = education[
        education["HFUDD"].isin([
            "H70 Lange videregående uddannelser, LVU",
            "H80 Ph.d. og forskeruddannelser"
        ])
    ].groupby("BOPOMR")["INDHOLD"].sum().reset_index()

    high.columns = ["Municipality", "High education"]

    edu = pd.merge(total, high, on="Municipality")
    edu["High education share"] = edu["High education"] / edu["Total"] * 100

    extension = pd.merge(
        latest[["Municipality", "Gini"]],
        edu[["Municipality", "High education share"]],
        on="Municipality"
    )

    return extension
