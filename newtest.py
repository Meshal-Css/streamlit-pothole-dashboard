import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from streamlit_chat import message
import openai

st.set_page_config(page_title="Pothole Detection", page_icon="🚧", layout="wide")


openai.api_key = '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'  # استبدل YOUR_API_KEY بمفتاحك

# Load YOLO model
model = YOLO("best (3).pt")

def apply_custom_css():
    st.markdown("""
        <style>
            /* General page settings */
            .main {
                background-color: #2E3440;  /* Darker blue background */
                color: #D8DEE9;  /* Light text */
            }

            /* Customize sidebar */
            .sidebar .sidebar-content {
                background-color: #3B4252;
                color: #D8DEE9;
            }

            /* Folium map styling */
            .leaflet-container {
                border: none;  /* Remove border */
            }

            .streamlit-folium {
                background-color: transparent;  /* Remove black background */
            }

            /* Button styling */
            .stButton > button {
                background-color: #5E81AC;  /* Light blue for buttons */
                color: white;
                border-radius: 8px;
                border: none;
                padding: 10px;
                transition: background-color 0.3s;
            }

            .stButton > button:hover {
                background-color: #81A1C1;  /* Hover effect */
            }

            /* Chat styling */
            .stTextInput > div > input {
                background-color: #D8DEE9;
                color: #2E3440;  /* Dark blue text */
                border: 2px solid #5E81AC;
                border-radius: 10px;
                padding: 8px;
            }

            /* Customize table */
            .stTable {
                background-color: #2E3440;
                color: #D8DEE9;
            }

            /* Chat message bubbles */
            .chat-message {
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
            }
            .user-message {
                background-color: #A3BE8C; /* User message color */
                color: black;
                text-align: right;
            }
            .bot-message {
                background-color: #81A1C1; /* Bot message color */
                color: black;
                text-align: left;
            }

            /* Add padding to chatbot */
            .chat-container {
                padding: 20px;
                border-radius: 10px;
                background-color: #3B4252;
                max-height: 400px;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)


def classify_pothole_severity(pothole_count):
    if pothole_count > 100:
        return "Severe"
    elif 50 <= pothole_count <= 100:
        return "High Risk"
    else:
        return "Low Risk"


def detect_potholes_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None, 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    pothole_count = 0  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        pothole_count += len(results[0].boxes)  
        result_frame = results[0].plot()  
        
        video_writer.write(result_frame)

    cap.release()
    video_writer.release()

    return output_video_path, pothole_count


def get_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    return response['choices'][0]['message']['content']


def main_page():
    st.title("Discovering Potholes on the Road")
    st.write("Upload a video to detect potholes.")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        temp_file.close()

        result_video_path, pothole_count = detect_potholes_video(temp_file.name)

        if result_video_path:
            st.video(result_video_path)
            st.success(f"Detected {pothole_count} potholes in the video.")

            pothole_severity = classify_pothole_severity(pothole_count)

            pothole_data = {
                "Pothole Count": [pothole_count],
                "Severity": [pothole_severity],
                "Pothole Location": ["Riyadh"]
            }
            df = pd.DataFrame(pothole_data)

            st.table(df)

            with open(result_video_path, "rb") as file:
                st.download_button(
                    label="Download Detected Potholes Video",
                    data=file,
                    file_name="detected_potholes.mp4",
                    mime="video/mp4"
                )
        else:
            st.write("No results found. Please check your model.")


def dashboard_page():
    st.title("Pothole Detection Dashboard")
    st.write("This page shows visualizations of detected potholes and cracks.")

    
    csv_file_path = "Pothole_ID,Location,Latitude,Longit.csv"  
    df = pd.read_csv(csv_file_path)

    
    m = folium.Map(location=[24.774265, 46.738586], zoom_start=12, tiles="CartoDB dark_matter")

    
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Pothole ID: {row['Pothole_ID']}<br>Location: {row['Location']}<br>Date Detected: {row['Date_Detected']}<br>Pothole Count: {row['Pothole_Count']}<br>Crack Count: {row['Crack_Count']}",
            icon=folium.Icon(color='red')
        ).add_to(m)

    
    st_folium(m, width=500, height=400)  

    
    st.subheader("Pothole Data")
    st.table(df)

    
    st.subheader("Pothole Count by Location")
    location_counts = df[['Location', 'Pothole_Count', 'Crack_Count']].groupby('Location').sum().reset_index()
    fig_bar = px.bar(location_counts, x='Location', y='Pothole_Count', title='Pothole Count by Location', color='Pothole_Count')
    st.plotly_chart(fig_bar)

    
    st.subheader("Crack Count by Location")
    fig_crack_bar = px.bar(location_counts, x='Location', y='Crack_Count', title='Crack Count by Location', color='Crack_Count')
    st.plotly_chart(fig_crack_bar)

    
    st.subheader("Comparison of Potholes and Cracks")
    comparison_df = location_counts.melt(id_vars='Location', value_vars=['Pothole_Count', 'Crack_Count'], var_name='Type', value_name='Count')
    fig_comparison = px.bar(comparison_df, x='Location', y='Count', color='Type', barmode='group', title='Comparison of Potholes and Cracks by Location')
    st.plotly_chart(fig_comparison)

    
    st.subheader("Neighborhoods with Most Potholes and Cracks")
    most_potholes = location_counts.sort_values(by='Pothole_Count', ascending=False).head(5)
    most_cracks = location_counts.sort_values(by='Crack_Count', ascending=False).head(5)
    
    st.write("Top 5 Neighborhoods with Most Potholes:")
    st.table(most_potholes[['Location', 'Pothole_Count']])

    st.write("Top 5 Neighborhoods with Most Cracks:")
    st.table(most_cracks[['Location', 'Crack_Count']])


def neighborhoods_page():
    st.title("Neighborhoods in Riyadh")
    st.write("Click on a neighborhood to see pothole details.")

    neighborhoods = {
        "Al-Narjis": {"location": [24.832592631493867, 46.682551601963375], "potholes": 30, "severity": "Low Risk"},
        "Al-Munsiyah": {"location": [24.839311029020966, 46.77422508429597], "potholes": 70, "severity": "High Risk"},
        "Qurtubah": {"location": [24.812424274752065, 46.75313865668322], "potholes": 120, "severity": "Severe"}
    }

    selected_neighborhood = st.selectbox("Select Neighborhood", options=list(neighborhoods.keys()))

    if selected_neighborhood:
        selected_info = neighborhoods[selected_neighborhood]
        st.write(f"Neighborhood: {selected_neighborhood}")
        st.write(f"Pothole Count: {selected_info['potholes']}")
        st.write(f"Severity: {selected_info['severity']}")

        
        m = folium.Map(location=selected_info["location"], zoom_start=15, tiles="CartoDB dark_matter")
        folium.Marker(location=selected_info["location"], popup=selected_neighborhood).add_to(m)
        st_folium(m, width=500, height=400)  

# Chatbot functionality
def chatbot():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = st.text_input("Ask me anything about pothole detection", key="user_input")
    
    if st.button("Send"):
        if user_input:
            output = get_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat messages
    with st.container():
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["generated"][i], key=f"generated_{i}", is_user=False)
            message(st.session_state['past'][i], key=f"past_{i}", is_user=True)

# Navigation
pages = {
    "Main Page": main_page,
    "Dashboard": dashboard_page,
    "Neighborhoods": neighborhoods_page,
    "Chatbot": chatbot
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", pages.keys())

apply_custom_css()

# Display 
pages[selected_page]()
