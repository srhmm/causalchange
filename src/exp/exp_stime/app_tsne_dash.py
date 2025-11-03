

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

from src.exp.exp_stime.data_hypparams import FLUX_OUT_PATH, PATH_PRE

df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_T_df.csv")
#df = pd.read_csv(PATH_PRE + PATH_RIVER_OUT + "tsne_output_2010.csv")
#print("Cols:", df.columns.tolist())

# Dash app
app = dash.Dash(__name__)
app.title = "t-SNE Interactive Viewer"

app.layout = html.Div([
    html.H2("t-SNE embedding for fluxnet", style={"textAlign": "center"}),

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
    fig.update_layout(template="plotly_white", title="Interactive t-SNE Embedding")


    return fig


if __name__ == "__main__":
    app.run(debug=True)

