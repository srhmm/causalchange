

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

from src.exp.exp_stime.data_hypparams import FLUX_OUT_PATH, PATH_PRE

df = pd.read_csv(PATH_PRE + FLUX_OUT_PATH + "tsne/tsne_years_col_T_df.csv")
#df = pd.read_csv(PATH_PRE + PATH_RIVER_OUT + "tsne_output_2010.csv")
print("Cols:", df.columns.tolist())

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
            html.Label("Trace Location"),
            dcc.Dropdown(
                options=[{"label": loc, "value": loc} for loc in sorted(df["loc"].unique())],
                id="trace-loc-dropdown",
                placeholder="Select location to trace...",
            ),
        ], style={"width": "30%", "display": "inline-block"}),
    ], style={"marginBottom": "20px"}),

    dcc.Graph(id="tsne-scatter", style={"height": "800px"}),

    html.Div([
        html.Label("Select Location for Yearly Trace"),
        dcc.Dropdown(
            id="loc-trace-dropdown",
            options=[{"label": loc, "value": loc} for loc in sorted(df["loc"].unique())],
            placeholder="Choose location to trace over years",
        ),
    ], style={"width": "40%", "marginTop": "20px"}),

    dcc.Graph(id="trace-graph", style={"height": "800px"}),
])

@app.callback(
    Output("tsne-scatter", "figure"),
    Input("year-dropdown", "value"),
    Input("month-dropdown", "value"),
    Input("trace-loc-dropdown", "value"),
)
def update_plot(selected_year, selected_month, selected_loc):
    # Base plot (light background)
    fig = px.scatter(
        df, x="tsne1", y="tsne2",
        opacity=0.1, color_discrete_sequence=["lightgray"],
        hover_data=["loc", "year", "month"],
    )
    fig.update_traces(marker=dict(size=4), showlegend=False)

    # Apply year/month filters (optional background filtering)
    mask = pd.Series(True, index=df.index)
    if selected_year is not None:
        mask &= (df["year"] == selected_year)
    if selected_month is not None:
        mask &= (df["month"] == selected_month)

    if selected_loc:
        loc_df = df[(df["loc"] == selected_loc)].copy()
        loc_df = loc_df.sort_values(by=["year", "month"])

        # Assign color by month (using Viridis or a custom list)
        month_colors = px.colors.sample_colorscale("Viridis", [m/11 for m in range(12)])
        color_map = {m: c for m, c in enumerate(month_colors)}
        loc_df["color"] = loc_df["month"].map(color_map)

        # Plot traced location: points + connecting line
        fig.add_scatter(
            x=loc_df["tsne1"], y=loc_df["tsne2"],
            mode="lines+markers",
            marker=dict(color=loc_df["color"], size=8),
            line=dict(color="black", width=1),
            name=f"{selected_loc}",
            text=[f"{y}-{m+1:02d}" for y, m in zip(loc_df["year"], loc_df["month"])],
            hoverinfo="text"
        )

    fig.update_layout(
        template="plotly_white",
        title=f"t-SNE Trajectory for Location {selected_loc}" if selected_loc else "t-SNE Embedding",
    )
    return fig


@app.callback(
    Output("trace-graph", "figure"),
    Input("loc-trace-dropdown", "value")
)
def update_trace_plot(selected_loc):
    if selected_loc is None:
        return px.scatter(title="Select a location to show trajectory")

    # Filter for the selected location
    loc_df = df[df["loc"] == selected_loc].copy()
    loc_df["month_label"] = loc_df["month"] + 1  # Make months 1–12

    # Set fixed categorical month labels for consistent coloring
    loc_df["month_name"] = pd.to_datetime(loc_df["month_label"], format="%m").dt.strftime("%b")

    # Define custom consistent colors for 12 months
    month_colorscale = px.colors.qualitative.Plotly  # or "Pastel", "Bold", etc.
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create main plot colored by month
    fig = px.scatter(
        loc_df,
        x="tsne1",
        y="tsne2",
        color="month_name",
        color_discrete_sequence=month_colorscale,
        category_orders={"month_name": month_order},
        hover_data=["year", "month_name"],
        title=f"t-SNE Monthly Trajectory for Location {selected_loc}",
    )

    # Add month centers with text labels
    centers = loc_df.groupby("month_name")[["tsne1", "tsne2"]].mean().reset_index()
    fig.add_scatter(
        x=centers["tsne1"],
        y=centers["tsne2"],
        mode="markers+text",
        text=centers["month_name"],
        textposition="top center",
        marker=dict(size=12, color="black", symbol="x"),
        name="Month Centers",
        showlegend=False
    )

    # Light background for other locations
    others = df[df["loc"] != selected_loc]
    fig.add_scatter(
        x=others["tsne1"],
        y=others["tsne2"],
        mode="markers",
        marker=dict(size=5, color="lightgray", opacity=0.1),
        name="Other locations",
        showlegend=False,
        hoverinfo="skip"
    )

    fig.update_layout(template="plotly_white")
    return fig



if __name__ == "__main__":
    app.run(debug=True)

