import fastf1
import plotly.graph_objs as go
import numpy as np
from dash import Dash, html, dcc, Output, Input, State
import pandas as pd

# Enable FastF1 cache
fastf1.Cache.enable_cache('./cache')

# Load 2023 Monaco race session
session = fastf1.get_session(2023, 'Monaco', 'R')
session.load()

laps = session.laps
drivers = laps['Driver'].unique()

# Start Dash App
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("F1 Race Dashboard"),

    html.Div([
        html.Label("Select Driver:"),
        dcc.Dropdown(
            id='driver-dropdown',
            options=[{'label': drv, 'value': drv} for drv in drivers],
            value=drivers[0]
        ),
        html.Br(),
        html.Label("Set Delay (in seconds):"),
        dcc.Input(id='delay-seconds', type='number', value=0, min=0, step=1),
    ]),

    html.Div([
        html.Label("Compare Laps (Delta Plot)", style={'marginTop': '15px'}),
        html.Div([
            html.Label("Lap A:"),
            dcc.Dropdown(id='lap-a'),
            html.Label("Lap B:"),
            dcc.Dropdown(id='lap-b'),
        ], style={'display': 'flex', 'gap': '10px'}),
    ]),
    dcc.Graph(id='delta-plot'),

    html.Div([
        html.Label("Compare Telemetry (Driver vs Driver)", style={'marginTop': '20px'}),

        html.Div([
            html.Label("Driver A:"),
            dcc.Dropdown(id='compare-driver-a', options=[{'label': d, 'value': d} for d in drivers], value=drivers[0]),

            html.Label("Driver B:"),
            dcc.Dropdown(id='compare-driver-b', options=[{'label': d, 'value': d} for d in drivers if d != drivers[0]], value=drivers[1]),

            html.Label("Lap #:"),
            dcc.Dropdown(id='compare-lap', value=1)
        ], style={'display': 'flex', 'gap': '10px'}),
    ]),

    dcc.Graph(id='compare-telemetry-plot'),

    html.Div([
        html.Label("Export Telemetry Data"),
        html.Button("Download CSV", id='download-btn'),
        dcc.Download(id="download-data")
    ], style={'marginTop': '20px'}),
    
    html.Div(id='lap-info'),
    html.Div(id='sector-times'),
    dcc.Interval(id='interval', interval=2000, n_intervals=0),
    dcc.Graph(id='telemetry-graph'),
    dcc.Graph(id='track-map')
])

@app.callback(
    [Output('lap-info', 'children'),
     Output('sector-times', 'children'),
     Output('telemetry-graph', 'figure'),
     Output('track-map', 'figure')],
    [Input('interval', 'n_intervals'),
     Input('driver-dropdown', 'value')],
    State('delay-seconds', 'value')
)
def update_dashboard(n_intervals, driver, delay_seconds):
    delay_seconds = delay_seconds or 0
    delay_laps = delay_seconds // 2  # assuming updates every 2 seconds
    lap_number = (n_intervals - delay_laps) % len(laps)
    if lap_number < 0:
        lap_number = 0

    driver_laps = laps.pick_drivers(driver).sort_values(by='LapNumber')
    all_laps_sorted = laps.sort_values(by='LapNumber')

    if lap_number >= len(driver_laps):
        lap_number = 0

    lap = driver_laps.iloc[lap_number]

    lap_time = lap['LapTime']
    lap_time_str = f"{lap_time.seconds % 3600 // 60}:{lap_time.seconds % 60:02}.{str(lap_time.microseconds)[:3]}"
    lap_info = f"Lap: {int(lap['LapNumber'])} | Lap Time: {lap_time_str}"

    def format_sector(t):
        if pd.isnull(t):
            return "N/A"
        return f"{t.seconds % 60:02}.{str(t.microseconds)[:3]}s"

    sector_times = (
        f"S1: {format_sector(lap['Sector1Time'])} | "
        f"S2: {format_sector(lap['Sector2Time'])} | "
        f"S3: {format_sector(lap['Sector3Time'])}"
    )

    # Tyre compound
    tyre_compound = lap.get('Compound', 'Unknown')
    tyre_text = f"Tyre Compound: {tyre_compound}"

    # Last pit stop
    pit_laps = driver_laps[driver_laps['PitOutTime'].notnull()]
    last_pit = "N/A"
    if not pit_laps.empty:
        last_pit_lap = pit_laps.iloc[-1]
        last_pit = f"Last Pit Stop: Lap {int(last_pit_lap['LapNumber'])}"

    # Gap to driver ahead (based on position and lap times)
    position = lap['Position']
    gap_text = "N/A"
    if position > 1:
        try:
            driver_time = lap['LapStartTime'] + lap['LapTime']
            ahead_driver_lap = all_laps_sorted[
                (all_laps_sorted['Position'] == position - 1) &
                (all_laps_sorted['LapNumber'] == lap['LapNumber'])
            ].iloc[0]
            ahead_time = ahead_driver_lap['LapStartTime'] + ahead_driver_lap['LapTime']
            gap = driver_time - ahead_time
            gap_secs = gap.total_seconds()
            gap_text = f"Gap to Driver Ahead: {gap_secs:.3f}s"
        except:
            pass

    # Weather data
    weather = session.weather_data
    weather_row = weather[weather['Time'] >= lap['LapStartTime']].iloc[0] if not weather.empty else None
    if weather_row is not None:
        air_temp = weather_row['AirTemp']
        track_temp = weather_row['TrackTemp']
        wind_speed = weather_row['WindSpeed']
        wind_dir = weather_row['WindDirection']
        weather_text = (
            f"Air Temp: {air_temp}°C | Track Temp: {track_temp}°C | "
            f"Wind: {wind_speed} km/h @ {wind_dir}°"
        )
    else:
        weather_text = "Weather data not available."

    combined = f"{lap_info} | {tyre_text} | {last_pit} | {gap_text} | {weather_text}"

    # Telemetry plot
    telemetry = None
    try:
        lap_telemetry = lap.get_telemetry()
        telemetry = go.Figure()

        telemetry.add_trace(go.Scatter(
            x=lap_telemetry['Distance'],
            y=lap_telemetry['Speed'],
            mode='lines',
            name='Speed (km/h)'
        ))

        telemetry.add_trace(go.Scatter(
            x=lap_telemetry['Distance'],
            y=lap_telemetry['Throttle'] * 100,
            mode='lines',
            name='Throttle (%)'
        ))

        telemetry.add_trace(go.Scatter(
            x=lap_telemetry['Distance'],
            y=lap_telemetry['Brake'].astype(int) * 100,
            mode='lines',
            name='Brake (%)'
        ))

        telemetry.update_layout(
            title=f"Telemetry: {driver} Lap {int(lap['LapNumber'])}",
            xaxis_title='Distance (m)',
            yaxis_title='Value',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
    except:
        telemetry = go.Figure().add_annotation(
            text="Telemetry data unavailable.",
            showarrow=False
        )

    # Track Map Plot (Polished)
    try:
        pos = lap.get_pos_data().reset_index(drop=True)

        # Normalize & center X and Y
        x = pos['X'] - pos['X'].mean()
        y = pos['Y'] - pos['Y'].mean()

        # Optional scaling (to make square-ish if needed)
        x = x / x.abs().max()
        y = y / y.abs().max()

        # Rotation (tune per track — using -30° here as an example)
        import numpy as np
        angle_deg = -30
        angle_rad = np.radians(angle_deg)

        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

        # Get car position (animate with intervals)
        progress_idx = min(n_intervals % len(pos), len(pos)-1)

        car_dot = go.Scatter(
            x=[x_rot.iloc[progress_idx]],
            y=[y_rot.iloc[progress_idx]],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=[driver],  # This shows driver code
            textposition="top center",
            textfont=dict(color='white', size=12),
            name='Car Position'
        )

        track_line = go.Scatter(
            x=x_rot,
            y=y_rot,
            mode='lines',
            line=dict(color='lightblue', width=2),
            name='Lap Path'
        )

        track_map = go.Figure(data=[track_line, car_dot])
        track_map.update_layout(
            title=f"Track Map: {driver} Lap {int(lap['LapNumber'])}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black'
        )
    except:
        track_map = go.Figure().add_annotation(
            text="Track data unavailable.",
            showarrow=False
        )
    return combined, sector_times, telemetry, track_map

@app.callback(
    Output('delta-plot', 'figure'),
    [Input('lap-a', 'value'),
     Input('lap-b', 'value'),
     Input('driver-dropdown', 'value')]
)
def update_delta_plot(lap_a_num, lap_b_num, driver):
    try:
        driver_laps = laps.pick_drivers(driver)

        lap_a_df = driver_laps[driver_laps['LapNumber'] == lap_a_num]
        lap_b_df = driver_laps[driver_laps['LapNumber'] == lap_b_num]
        if lap_a_df.empty or lap_b_df.empty:
            raise ValueError("One of the laps is missing")

        lap_a = lap_a_df.iloc[0]
        lap_b = lap_b_df.iloc[0]

        tel_a = lap_a.get_telemetry().add_distance()
        tel_b = lap_b.get_telemetry().add_distance()

        interp_b = np.interp(tel_a['Distance'], tel_b['Distance'], tel_b['Speed'])
        delta = tel_a['Speed'] - interp_b

        fig = go.Figure()

        # Lap A (blue)
        fig.add_trace(go.Scatter(
            x=tel_a['Distance'],
            y=tel_a['Speed'],
            mode='lines',
            name=f"Lap {lap_a_num} Speed",
            line=dict(color='blue')
        ))

        # Lap B (red)
        fig.add_trace(go.Scatter(
            x=tel_b['Distance'],
            y=tel_b['Speed'],
            mode='lines',
            name=f"Lap {lap_b_num} Speed",
            line=dict(color='red')
        ))

        # Delta (orange)
        fig.add_trace(go.Scatter(
            x=tel_a['Distance'],
            y=delta,
            mode='lines',
            name=f"Delta (A - B)",
            line=dict(color='orange', dash='dash')
        ))

        fig.update_layout(
            title=f"Delta Plot: {driver} Lap {lap_a_num} vs Lap {lap_b_num}",
            xaxis_title="Distance (m)",
            yaxis_title="Speed (km/h)",
            height=400,
            margin=dict(l=30, r=30, t=50, b=30)
        )

        return fig

    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Delta data unavailable: {str(e)}",
            showarrow=False
        )

@app.callback(
    [Output('lap-a', 'options'),
     Output('lap-b', 'options'),
     Output('lap-a', 'value'),
     Output('lap-b', 'value')],
    [Input('driver-dropdown', 'value')]
)
def update_lap_dropdowns(driver):
    driver_laps = laps.pick_drivers(driver)
    lap_nums = sorted(driver_laps['LapNumber'].unique())

    options = [{'label': f"Lap {lap}", 'value': lap} for lap in lap_nums]

    # Pick two distinct laps (prefer last two if possible)
    if len(lap_nums) >= 2:
        lap_a_val = lap_nums[-2]
        lap_b_val = lap_nums[-1]
    else:
        lap_a_val = lap_nums[0]
        lap_b_val = lap_nums[0]

    return options, options, lap_a_val, lap_b_val

@app.callback(
    Output('compare-lap', 'options'),
    [Input('compare-driver-a', 'value'),
     Input('compare-driver-b', 'value')]
)
def update_compare_laps(driver_a, driver_b):
    laps_a = set(laps.pick_drivers(driver_a)['LapNumber'])
    laps_b = set(laps.pick_drivers(driver_b)['LapNumber'])
    shared = sorted(list(laps_a & laps_b))
    return [{'label': f"Lap {l}", 'value': l} for l in shared]

@app.callback(
    Output('compare-telemetry-plot', 'figure'),
    [Input('compare-driver-a', 'value'),
     Input('compare-driver-b', 'value'),
     Input('compare-lap', 'value')]
)
def update_driver_compare(driver_a, driver_b, lap_num):
    try:
        lap_a = laps.pick_drivers(driver_a)[laps['LapNumber'] == lap_num].iloc[0]
        lap_b = laps.pick_drivers(driver_b)[laps['LapNumber'] == lap_num].iloc[0]

        tel_a = lap_a.get_telemetry().add_distance()
        tel_b = lap_b.get_telemetry().add_distance()

        fig = go.Figure()

        for data, label, color in [
            (tel_a, driver_a, 'blue'),
            (tel_b, driver_b, 'red')
        ]:
            fig.add_trace(go.Scatter(
                x=data['Distance'],
                y=data['Speed'],
                name=f"{label} Speed",
                line=dict(color=color)
            ))
            fig.add_trace(go.Scatter(
                x=data['Distance'],
                y=data['Throttle'] * 100,
                name=f"{label} Throttle (%)",
                line=dict(color=color, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=data['Distance'],
                y=data['Brake'].astype(int) * 100,
                name=f"{label} Brake (%)",
                line=dict(color=color, dash='dash')
            ))

        fig.update_layout(
            title=f"{driver_a} vs {driver_b} - Lap {lap_num} Telemetry",
            xaxis_title="Distance (m)",
            yaxis_title="Value",
            height=500,
            margin=dict(l=30, r=30, t=50, b=30)
        )

        return fig

    except Exception as e:
        return go.Figure().add_annotation(text=f"Telemetry data unavailable: {str(e)}", showarrow=False)

@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("driver-dropdown", "value"),
    prevent_initial_call=True
)
def export_driver_data(n_clicks, driver):
    print(f"Download requested for driver: {driver}")
    try:
        driver_laps = laps.pick_drivers(driver)
        all_data = []

        for _, lap_row in driver_laps.iterrows():
            try:
                lap = lap_row
                print(f"Processing Lap {lap['LapNumber']}")
                tel = lap.get_telemetry().add_distance()
                tel['LapNumber'] = lap['LapNumber']
                tel['Driver'] = driver
                all_data.append(tel)
            except Exception as e:
                print(f"Lap {lap['LapNumber']} failed: {e}")
                continue

        if not all_data:
            raise ValueError("No telemetry data available")

        df = pd.concat(all_data)
        print(f"Exporting {len(df)} rows for {driver}")
        return dcc.send_data_frame(df.to_csv, f"{driver}_telemetry.csv", index=False)

    except Exception as e:
        print(f"Export failed: {str(e)}")
        return dcc.send_string(f"Export failed: {str(e)}", filename="error.txt")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)