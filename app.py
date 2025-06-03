import dash
from dash import dcc, html, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras import layers, models
import plotly.graph_objects as go


twolves_data = pd.read_csv("twolves_data.csv")
nyk_data = pd.read_csv("nyk_data.csv")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def draw_court_shapes(fig):
    # Hoop
    fig.add_shape(type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, line=dict(color="black"))
    # Backboard
    fig.add_shape(type="line", x0=-30, y0=-7.5, x1=30, y1=-7.5, line=dict(color="black"))
    # Outer box
    fig.add_shape(type="rect", x0=-80, y0=-47.5, x1=80, y1=142.5, line=dict(color="black"))
    # Inner box
    fig.add_shape(type="rect", x0=-60, y0=-47.5, x1=60, y1=142.5, line=dict(color="black"))
    # Free throw circle
    fig.add_shape(type="circle", x0=-60, y0=82.5, x1=60, y1=202.5, line=dict(color="black"))
    # Restricted area
    fig.add_shape(type="path", path="M -40 0 A 40 40 0 0 1 40 0", line=dict(color="black"))
    # Corner threes
    fig.add_shape(type="line", x0=-220, y0=-47.5, x1=-220, y1=92.5, line=dict(color="black"))
    fig.add_shape(type="line", x0=220, y0=-47.5, x1=220, y1=92.5, line=dict(color="black"))
    # Three-point arc
    theta = np.linspace(np.radians(22), np.radians(158), 200)
    arc_x = 237.5 * np.cos(theta)
    arc_y = 237.5 * np.sin(theta)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines', line=dict(color='black'), showlegend=False))




def simulate_matchup_with_probabilities(shooter_name, defender_name):
    shooter_df = twolves_data[twolves_data['PLAYER_NAME'] == shooter_name].copy()
    defender_row = nyk_data[nyk_data['PLAYER_NAME'] == defender_name].copy()
    if shooter_df.empty or defender_row.empty:
        return pd.DataFrame()

    defender_metrics = ['Def. Rebound %', 'Steal %', 'Block %', 'Def. Win Shares', 'Def. Box +/-']
    for metric in defender_metrics:
        defender_row[metric] = pd.to_numeric(defender_row[metric], errors='coerce')
    defender_stats = defender_row[defender_metrics].mean()

    for col in ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']:
        shooter_df[col] = pd.to_numeric(shooter_df[col], errors='coerce')

    shooter_df['SHOT_TYPE'] = shooter_df['SHOT_TYPE'].astype(str)
    shooter_df['opp_Def. Rebound %'] = defender_stats['Def. Rebound %']
    shooter_df['opp_Steal %'] = defender_stats['Steal %']
    shooter_df['opp_Block %'] = defender_stats['Block %']
    shooter_df['opp_Def. Win Shares'] = defender_stats['Def. Win Shares']
    shooter_df['opp_Def. Box +/-'] = defender_stats['Def. Box +/-']

    shooter_df['angle_from_basket'] = np.degrees(np.arctan2(shooter_df['LOC_Y'], shooter_df['LOC_X']))
    shooter_df['POINTS'] = shooter_df.apply(
        lambda row: 3 if '3PT' in row['SHOT_TYPE'] and row['SHOT_MADE_FLAG'] == 1
        else 2 if '2PT' in row['SHOT_TYPE'] and row['SHOT_MADE_FLAG'] == 1
        else 0, axis=1)
    shooter_df['home_game'] = shooter_df['HTM'].apply(lambda x: 1 if x == 'Minnesota Timberwolves' else 0)
    shooter_df['is_corner_three'] = shooter_df.apply(
        lambda row: 1 if (abs(row['LOC_X']) > 220 and row['LOC_Y'] < 100 and '3PT' in str(row['SHOT_TYPE'])) else 0, axis=1)
    shooter_df['is_midrange'] = shooter_df['SHOT_DISTANCE'].apply(lambda d: 1 if 10 <= d <= 18 else 0)

    shooter_df['COURT_ZONE'] = shooter_df['SHOT_ZONE_BASIC'] + ' – ' + shooter_df['SHOT_ZONE_AREA']
    zone_dummies = pd.get_dummies(shooter_df['SHOT_ZONE_BASIC'], prefix='zone')
    shooter_df = pd.concat([shooter_df, zone_dummies], axis=1)

    zone_cols = [
        'zone_Above the Break 3', 'zone_Backcourt', 'zone_In The Paint (Non-RA)',
        'zone_Left Corner 3', 'zone_Mid-Range', 'zone_Restricted Area', 'zone_Right Corner 3'
    ]
    for col in zone_cols:
        if col not in shooter_df:
            shooter_df[col] = 0

    shooter_df['dist_0-5'] = shooter_df['SHOT_DISTANCE'].apply(lambda x: 1 if 0 <= x < 5 else 0)
    shooter_df['dist_5-10'] = shooter_df['SHOT_DISTANCE'].apply(lambda x: 1 if 5 <= x < 10 else 0)

    numeric_features = ['LOC_X', 'LOC_Y', 'angle_from_basket', 'POINTS', 'opp_Def. Rebound %', 'opp_Steal %', 'opp_Block %', 'opp_Def. Win Shares', 'opp_Def. Box +/-']
    binary_features = ['home_game', 'is_corner_three', 'is_midrange', 'dist_0-5', 'dist_5-10']
    categorical_features = ['SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE']

    required_cols = numeric_features + binary_features + zone_cols + categorical_features + ['SHOT_MADE_FLAG']
    shooter_df = shooter_df.dropna(subset=required_cols)

    X = shooter_df[numeric_features + binary_features + zone_cols + categorical_features]
    y = shooter_df['SHOT_MADE_FLAG'].astype('float32')

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features + binary_features + zone_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)

    model = models.Sequential([
        layers.Input(shape=(X_processed.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_processed, y, epochs=25, batch_size=32, verbose=0)

    probabilities = model.predict(X_processed).flatten()
    shooter_df['predicted_fg_pct'] = probabilities

    return shooter_df


app.layout = dbc.Container([
    html.H2("NBA Matchup FG% Simulation", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Shooter (Timberwolves):"),
            dcc.Dropdown(
                id='shooter-dropdown',
                options=[{'label': name, 'value': name} for name in sorted(twolves_data['PLAYER_NAME'].unique())],
                value=twolves_data['PLAYER_NAME'].unique()[0]
            )
        ]),
        dbc.Col([
            html.Label("Select Defender (Knicks):"),
            dcc.Dropdown(
                id='defender-dropdown',
                options=[{'label': name, 'value': name} for name in sorted(nyk_data['PLAYER_NAME'].unique())],
                value=nyk_data['PLAYER_NAME'].unique()[0]
            )
        ])
    ], className="mb-4"),
    dcc.Graph(id='shot-chart')
])


@app.callback(
    Output('shot-chart', 'figure'),
    Input('shooter-dropdown', 'value'),
    Input('defender-dropdown', 'value')
)
def update_graph(shooter_name, defender_name):
    shooter_df = simulate_matchup_with_probabilities(shooter_name, defender_name)
    if shooter_df.empty:
        return px.scatter(title="No Data Found")

    shooter_df['COURT_ZONE'] = shooter_df['SHOT_ZONE_BASIC'] + ' – ' + shooter_df['SHOT_ZONE_AREA']
    zone_coords = {
        'Restricted Area – Center(C)': (0, 40),
        'In The Paint (Non-RA) – Center(C)': (0, 100),
        'In The Paint (Non-RA) – Left Side(L)': (-100, 100),
        'In The Paint (Non-RA) – Right Side(R)': (100, 100),
        'Mid-Range – Left Side(L)': (-150, 125),
        'Mid-Range – Left Side Center(LC)': (-75, 150),
        'Mid-Range – Center(C)': (0, 175),
        'Mid-Range – Right Side Center(RC)': (75, 150),
        'Mid-Range – Right Side(R)': (150, 125),
        'Left Corner 3 – Left Side(L)': (-220, 40),
        'Right Corner 3 – Right Side(R)': (220, 40),
        'Above the Break 3 – Left Side Center(LC)': (-75, 250),
        'Above the Break 3 – Center(C)': (0, 250),
        'Above the Break 3 – Right Side Center(RC)': (75, 250),
        'Backcourt – Back Court(BC)': (0, -100),
        'Above the Break 3 – Back Court(BC)': (0, -50)
    }

    zone_pred = shooter_df.groupby('COURT_ZONE').agg(
        predicted_fg_pct=('predicted_fg_pct', 'mean'),
        count=('predicted_fg_pct', 'count')
    ).reset_index()

    zone_pred['x'] = zone_pred['COURT_ZONE'].map(lambda z: zone_coords.get(z, (None, None))[0])
    zone_pred['y'] = zone_pred['COURT_ZONE'].map(lambda z: zone_coords.get(z, (None, None))[1])
    zone_pred = zone_pred.dropna(subset=['x', 'y'])

    fig = px.scatter(
    zone_pred, x='x', y='y', size='count', color='predicted_fg_pct',
    color_continuous_scale='Blues',
    hover_data={'predicted_fg_pct': ':.2%'}, 
    labels={'predicted_fg_pct': 'Pred FG%'},
    title=f"Predicted FG% by Zone for {shooter_name} vs {defender_name}"
)
    fig.update_traces(
    hovertemplate='<b>FG%:</b> %{marker.color:.2%}<extra></extra>'
)

    draw_court_shapes(fig)

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        plot_bgcolor='#F5F5F5',
        height=600
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)
