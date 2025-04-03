import streamlit as st
import pickle
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Load the model from a pickle file
with open('../x_Exported_Model_housing-model_GradientBoostingRegressor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    # print(dir(model))
    print('Model Class: ',model.__class__)
    print('Model Inputs', model.n_features_in_)
    # print('Model Outputs', model.n_outputs_)

# Define the path to the data file
DATA_FILE = 'predictions_data.csv'

# Create the data file if it doesn't exist
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=['Reference', 'Inputs', 'Prediction', 'Feedback'])
    df = df.astype({
        'Reference': 'int64',
        'Inputs': 'object',
        'Prediction': 'object',
        'Feedback': 'object'
    })
    df.to_csv(DATA_FILE, index=False)

def get_next_reference():
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        return 1
    else:
        return df['Reference'].max() + 1

def save_to_data_file(reference, inputs, prediction, feedback=''):
    df = pd.read_csv(DATA_FILE)
    new_row = pd.DataFrame.from_records([{
        'Reference': reference,
        'Inputs': inputs,
        'Prediction': prediction,
        'Feedback': feedback
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def send_feedback(reference, feedback):
    # This is where you send feedback to another Streamlit API endpoint
    # For now, we just update the local file
    df = pd.read_csv(DATA_FILE)
    df = df.astype({
        'Reference': 'int64',
        'Inputs': 'object',
        'Prediction': 'object',
        'Feedback': 'object'
    })
    df.loc[df['Reference'] == reference, 'Feedback'] = feedback
    print(df.loc[df['Reference'] == reference])
    df.to_csv(DATA_FILE, index=False)


def plotly_map(df, latlng_cols=('lat','lng'), z=None, custom_data_cols=[], custom_text=[], center_dict=dict(lat=13.6, lon=100.4), zoom=5, WRITE=False, WRITE_FN=None):
    """ 
    @WRITE_FN - do not include extension - i.e. `.png` or `.html`, as both files will be written.
    Docs:   https://plotly.com/python-api-reference/generated/plotly.express.density_mapbox.html
            https://plotly.com/python/mapbox-density-heatmaps/
    """
    pio.templates.default = 'plotly_white' # 'plotly_dark'
    fig = px.density_mapbox(df, 
                            lat=latlng_cols[0], 
                            lon=latlng_cols[1], 
                            z=z,
                            radius=5,
                            center=center_dict, zoom=zoom,
                            mapbox_style=["open-street-map",'carto-darkmatter'][0],
                            custom_data=custom_data_cols,
                           )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    if custom_text:
        fig.update_traces(
            hovertemplate="<br>".join(custom_text)
        )
    return fig

def dashboard(dft):


    # Convert 'Date' to datetime format
    dft['Date'] = pd.to_datetime(dft['Date'], format='%d/%m/%Y')

    # Create a Streamlit app
    st.title('Housing Data Visualization Dashboard')

    # Sidebar for filters
    st.sidebar.header('Filters')

    # Filter by Suburb
    suburb_options = dft['Suburb'].unique()
    selected_suburbs = st.sidebar.multiselect('Select Suburbs', suburb_options, default=suburb_options)

    # Filter by Type
    type_options = dft['Type'].unique()
    selected_types = st.sidebar.multiselect('Select Property Types', type_options, default=type_options)

    # Filter data based on selections
    filtered_dft = dft[(dft['Suburb'].isin(selected_suburbs)) & (dft['Type'].isin(selected_types))]

    # Plot Price Distribution
    st.subheader('Housing Market Map')
    dfm = filtered_dft.dropna()
    fig = plotly_map(dfm, 
            latlng_cols=('Lattitude','Longtitude'), 
            z='Price',
            custom_data_cols=['CouncilArea',
                                'Distance',
                                'Landsize',
                                'BuildingArea',
                                'Rooms',
                                'Bathroom',
                                'Price'
                            ], 
            custom_text=['Area: %{customdata[0]}',
                            'Distance: %{customdata[1]}',
                            'LS / BA: %{customdata[2]}/%{customdata[3]}',
                            'Rm / Br: %{customdata[4]}/%{customdata[5]}',
                            'Price AUD-$: %{customdata[6]:.,1f}'
                                ],
            center_dict=dict(lat=-37.814, lon=144.963),
            zoom=9
            )
    st.plotly_chart(fig)

    # Plot Price Distribution
    st.subheader('Price Distribution')
    fig = px.histogram(filtered_dft, x='Price', nbins=30, title='Price Distribution', marginal='box')
    st.plotly_chart(fig)

    # Plot Price vs. Rooms
    st.subheader('Price vs. Rooms')
    fig = px.scatter(filtered_dft, x='Rooms', y='Price', title='Price vs. Number of Rooms')
    st.plotly_chart(fig)

    # Plot Price by Property Type
    st.subheader('Average Price by Property Type')
    fig = px.bar(filtered_dft.groupby('Type', as_index=False).agg({'Price': 'mean'}), 
                x='Type', y='Price', title='Average Price by Property Type')
    st.plotly_chart(fig)

    # Plot Price vs. Distance to CBD
    st.subheader('Price vs. Distance to CBD')
    fig = px.scatter(filtered_dft, x='Distance', y='Price', title='Price vs. Distance to CBD')
    st.plotly_chart(fig)

    # Plot Price by Year Built
    st.subheader('Average Price by Year Built')
    fig = px.bar(filtered_dft.groupby('YearBuilt', as_index=False).agg({'Price': 'mean'}), 
                x='YearBuilt', y='Price', title='Average Price by Year Built')
    st.plotly_chart(fig)

    # Plot Correlation Heatmap
    st.subheader('Correlation Heatmap')
    correlation_matrix = filtered_dft[['Price', 'Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt']].corr()
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.columns, colorscale='Viridis'))
    fig.update_layout(title='Correlation Heatmap')
    st.plotly_chart(fig)
        

def plot_prediction(dft, prediction, target_col='Price'):
    df_sorted = dft.sort_values(by=target_col).reset_index(drop=True)
    fig = go.Figure()

    # Add bars to the plot
    fig.add_trace(go.Bar(
        x=df_sorted.index,
        y=df_sorted[target_col],
        marker=dict(color='red'),
        name='Price'
    ))

    # Add a vertical dashed line
    fig.add_hline(
        y=prediction,#df_sorted[df_sorted[target_col] == prediction].index[0] if prediction in df_sorted[target_col].values else None,
        line=dict(color='red', dash='dash'),
        name='Prediction',
    )

    # Update layout for better appearance
    fig.update_layout(
        title=f'{target_col} - Prediction at Red Line',
        xaxis_title='Index',
        yaxis_title=target_col,
        showlegend=False
    )
    st.plotly_chart(fig)

def main():
    st.title('Housing Pricing & Prediction App:')

    with open('../files/melb_data_housing.md', 'r') as file:
        markdown_content = file.read()
    with st.expander("See dataset explanation"):
        st.markdown(markdown_content)

    # Load the dataset
    df_h = pd.read_csv('../files/melb_data.csv').drop('Unnamed: 0', axis=1)
    with st.expander("Dataset Visualizations"):
        dashboard(df_h)
    target = 'Price'
    # columns = set(df_h.columns)-set([target])
    columns = ['Distance', 'Landsize', 'BuildingArea', 'Rooms', 'Bathroom']
    dtypes = df_h.dtypes
    # Initialize a dictionary to hold the user inputs
    user_inputs = {}

    with st.expander("Make Prediction from Inputs:"):
        st.subheader('Make Prediction from Inputs:')
        # Create input fields based on column data types
        for column in columns:
            dtype = dtypes[column]
            if dtype == 'object':  # Categorical or string
                user_inputs[column] =  st.select_slider(column,
                                            options= df_h[column].unique().tolist(),
                                            value=df_h[column].mode()[0]
                                            )
            elif dtype == 'int64':  # Integer
                user_inputs[column] = st.number_input(column, format='%d',
                                            value=df_h[column].mean())
            elif dtype == 'float64':  # Float
                user_inputs[column] = st.number_input(column, format='%.2f',
                                            value=df_h[column].median())
            elif dtype == 'bool':  # Boolean
                user_inputs[column] = st.checkbox(column)
            elif dtype == 'datetime64[ns]':
                user_inputs[column] = st.select_slider(column,
                                            options= df_h[column].astype(str).unique().tolist(),
                                            value=df_h[column].astype(str).mode()[0], 
                                            )
                                        # st.date_input(column,
                                        #     value=df_h[column].mean())
            else:
                st.write(f"Unhandled data type for column {column}: {dtype}")

        # Display the collected inputs
        st.write("Housing Data Inputs:")
        st.write(user_inputs)
    

        if st.button('Submit'):
            # Prepare input data
            inputs = [user_inputs[c] if dtypes[c] != 'datetime64[ns]' else user_inputs[c][0:3] for c in columns]
            # Make prediction
            prediction = model.predict([inputs])[0]
            # Generate reference number
            reference = get_next_reference()
            st.session_state.current_ref_number = reference
            # Save to data file
            save_to_data_file(reference, inputs, prediction)
            # Display prediction
            st.write(f"{target} Prediction: {prediction}")
            plot_prediction(df_h, prediction)
            
            # Feedback options
            st.write(f"Was this prediction helpful?")
            
            # sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
            # selected = st.feedback("thumbs")
            # if selected is not None:
            #     st.markdown(f"You selected: {sentiment_mapping[selected]}")
            #     st.write(f"You selected: {sentiment_mapping[selected]}")
            sentiment_mapping = ["one", "two", "three", "four", "five"]
            selected = st.feedback("stars")
            if selected is not None:
                st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
            
            
            # sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
            # feedback = st.feedback("thumbs", on_change=handle_feedback)
            # if feedback is not None:
                # if st.session_state.get('feedback_received') is not None:
                    # st.write(f"Feedback received: {st.session_state.feedback_received}")

                # Display additional information if feedback was received
                # if st.session_state.get('feedback_received') is not None:
                    # st.write(f"Thank you for your feedback! You said: {st.session_state.feedback_received}")
            #     st.markdown(f"You selected: {sentiment_mapping[feedback]}")
            #     print(st.session_state.current_ref_number, f"You selected: {sentiment_mapping[feedback]}")
                send_feedback(st.session_state.current_ref_number, f"You selected: {sentiment_mapping[selected]}")
            

if __name__ == "__main__":
    main()
    # Self-Driving Car Demo example: https://github.com/streamlit/demo-self-driving
    # pip install --upgrade streamlit opencv-python
    # streamlit run https://raw.githubusercontent.com/streamlit/demo-self-driving/master/streamlit_app.py
