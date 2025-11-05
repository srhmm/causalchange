import argparse
import sys

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

from src.exp.exp_stime.data_hypparams import FLUX_OUT_PATH, PATH_PRE, PATH_RIVER_OUT

ap = argparse.ArgumentParser("tsne")
ap.add_argument("-m", "--mode", default=1, help=f"colors (0 for clusters, 1 for P, 2 for R, 3 for T, 4 for VPD, 5 for NEE, 6 for LE, 7 for GPP)", type=int)
argv = sys.argv[1:]
nmsp = ap.parse_args(argv)

if nmsp.mode == 0:
    df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_cluster_df.csv")
    dfname = 'fluxnet, colored by cluster/region'
elif nmsp.mode == 1:
    df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_P_df.csv")
    dfname = 'fluxnet, colored by P (precipitation)'
elif nmsp.mode == 2:
    df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_R_df.csv")
    dfname = 'fluxnet, colored by R (radiation)'
elif nmsp.mode == 3:
    df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_T_df.csv")
    dfname = 'fluxnet, colored by T (temperature)'
elif nmsp.mode == 4:
    df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_VPD_df.csv")
    dfname = 'fluxnet, colored by VPD'
elif nmsp.mode == 5:
    df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_NEE_df.csv")
    dfname = 'fluxnet, colored by NEE'
elif nmsp.mode == 6:
    df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_LE_df.csv")
    dfname = 'fluxnet, colored by LE'
elif nmsp.mode == 7:
    df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_GPP_df.csv")
    dfname = 'fluxnet, colored by GPP'
elif nmsp.mode == 8:
    df = pd.read_csv(PATH_PRE + PATH_RIVER_OUT + "tsne_output_2010_colorby_edgePQ.csv")
    dfname = 'river'
else:
    raise ValueError(f"{nmsp.mode} invalid color")
#print("Cols:", df.columns.tolist())

# Dash app
app = dash.Dash(__name__)
app.title = "t-SNE embedding"

app.layout = html.Div([
    html.H2(f"t-SNE embedding for {dfname}", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Label("Select Year"),
            dcc.Dropdown(
                options=[{"label": str(y), "value": y} for y in sorted(df["year"].unique())],
                id="year-dropdown",
                placeholder="All Years",
            ),
        ], style={"width": "30%", "display": "inline-block"}),

        html.Div([
            html.Label("Select Month"),
            dcc.Dropdown(
                options=[{"label": f"{m+1:02d}", "value": m} for m in range(12)],
                id="month-dropdown",
                placeholder="All Months",
            ),
        ], style={"width": "30%", "display": "inline-block"}),

        html.Div([
            html.Label("Select Location"),
            dcc.Dropdown(
                options=[{"label": loc, "value": loc} for loc in sorted(df["loc"].unique())],
                id="loc-dropdown",
                placeholder="All Locations",
            ),
        ], style={"width": "30%", "display": "inline-block"}),
    ], style={"marginBottom": "20px"}),

    dcc.Graph(id="tsne-scatter", style={"height": "800px"}),
])


@app.callback(
    Output("tsne-scatter", "figure"),
    Input("year-dropdown", "value"),
    Input("month-dropdown", "value"),
    Input("loc-dropdown", "value"),
)




def update_plot(selected_year, selected_month, selected_loc):


    fig = px.scatter(
        df,
        x="tsne1",
        y="tsne2",
        color_discrete_sequence=["lightgray"],
        opacity=0.3,
        hover_data=["loc", "year", "month"],
    )
    fig.update_traces(marker=dict(size=5), showlegend=False)

    # Highlight subset
    mask = pd.Series(True, index=df.index)
    if selected_year is not None:
        mask &= (df["year"] == selected_year)
    if selected_month is not None:
        mask &= (df["month"] == selected_month)
    if selected_loc is not None:
        mask &= (df["loc"] == selected_loc)

    selected_df = df[mask]

    if not selected_df.empty:
        fig.add_trace(
            px.scatter(
                selected_df,
                x="tsne1",
                y="tsne2",
                color="color_value",  # use color_value column
                color_continuous_scale="RdBu",#"Viridis",
                opacity=0.9,
                hover_data=["loc", "year", "month", "color_value"]
            ).data[0]  # grab the first trace from the returned figure
        )
    fig.update_layout(coloraxis_colorbar=dict(title="color_value"))
    #fig.add_scatter(
    #    x=selected_df["tsne1"],
    #    y=selected_df["tsne2"],
    #    mode="markers",
    #    marker=dict(size=10, color="red", opacity=0.9),
    #    name="Selected",
    #)
    fig.update_layout(template="plotly_white", title=f"ft-SNE embedding {dfname}")


    return fig


if __name__ == "__main__":
    app.run(debug=True)

