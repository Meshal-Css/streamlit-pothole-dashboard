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
import random
import numpy as np
from PIL import Image

st.set_page_config(page_title="Pothole Detection", page_icon="🚧", layout="wide")

openai.api_key = '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'  

# Load YOLO model
model = YOLO("best (3).pt")

faq_df = pd.read_csv('pothole_detection_faq.csv')  
faq_df.columns = faq_df.columns.str.strip() 

pothole_data = pd.read_csv('Pothole_ID,Location,Latitude,Longit.csv')
pothole_data.columns = pothole_data.columns.str.strip()  

def apply_custom_css():
    st.markdown("""
        <style>
            /* خلفية الصورة */
            body {
                background-image: url('C:/Users/MSI1/Documents/VS/image ba.jpg'); /* استبدل بالمسار الخاص بالصورة */
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }

            /* إعدادات الصفحة العامة */
            .main {
                background-color: rgba(46, 52, 64, 0.8);
                color: #D8DEE9;
            }

            /* تخصيص الشريط الجانبي */
            .sidebar .sidebar-content {
                background-color: rgba(59, 66, 82, 0.8);
                color: #D8DEE9;
            }

            /* تخصيص زر */
            .stButton > button {
                background-color: #5E81AC;  
                color: white;
                border-radius: 8px;
                border: none;
                padding: 10px;
                transition: background-color 0.3s;
            }

            .stButton > button:hover {
                background-color: #81A1C1;  
            }

            /* تخصيص الرسائل في الدردشة */
            .chat-message {
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
            }
            .user-message {
                background-color: #A3BE8C; 
                color: black;
                text-align: right;
            }
            .bot-message {
                background-color: #81A1C1; 
                color: black;
                text-align: left;
            }

            /* إضافة padding للدردشة */
            .chat-container {
                padding: 20px;
                border-radius: 10px;
                background-color: rgba(59, 66, 82, 0.8);
                max-height: 400px;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)

def classify_surface_condition(prediction):
    if prediction == 0:
        return "Pothole detected"
    elif prediction == 1:
        return "Cracks detected"
    else:
        return random.choice(["Pothole detected", "Cracks detected"])

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

def detect_potholes_image(image):
    # تحويل الصورة إلى RGB
    image = image.convert("RGB")
    img = np.array(image)
    results = model(img)
    
    pothole_count = len(results[0].boxes)  # عدد الحفر المكتشفة
    result_image = results[0].plot()  # رسم النتائج على الصورة

    return result_image, pothole_count

    return result_image, pothole_count

def get_response(user_input):
    answer = faq_df.loc[faq_df['question'].str.lower() == user_input.lower(), 'answer']
    if not answer.empty:
        return answer.values[0]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    return response['choices'][0]['message']['content']

def main_page():
    st.title("Discovering Potholes on the Road")
    st.write("Upload a video and an image to detect potholes.")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_video is not None:
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_file.write(uploaded_video.read())
        temp_video_file.close()

        result_video_path, pothole_count = detect_potholes_video(temp_video_file.name)

        if result_video_path:
            st.video(result_video_path)
            st.success(f"Detected {pothole_count} potholes in the video.")

            pothole_severity = classify_surface_condition(pothole_count)

            pothole_data = {
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

    if uploaded_image is not None:
        # قراءة الصورة باستخدام PIL
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # معالجة الصورة للتنبؤ
        result_image, pothole_count = detect_potholes_image(image)

        # عرض الصورة مع النتائج
        st.image(result_image, caption="Detected Potholes", use_column_width=True)
        st.success(f"Detected {pothole_count} potholes in the image.")

def dashboard_page():
    st.title("Pothole Detection Dashboard")
    st.write("This page shows visualizations of detected potholes and cracks.")

    m = folium.Map(location=[24.774265, 46.738586], zoom_start=13, tiles="CartoDB dark_matter")  # تقليل مستوى التكبير

    for _, row in pothole_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Pothole ID: {row['Pothole_ID']}<br>Location: {row['Location']}<br>Date Detected: {row['Date_Detected']}<br>Pothole Count: {row['Pothole_Count']}<br>Crack Count: {row['Crack_Count']}",
            icon=folium.Icon(color='red')
        ).add_to(m)

    # تقليل حجم الخريطة
    st_folium(m, width=900, height=800)  # تغيير الأبعاد هنا

    st.subheader("Pothole Data")
    st.table(pothole_data)

    location_counts = pothole_data.groupby('Location')[['Pothole_Count', 'Crack_Count']].sum().reset_index()
    fig_bar = px.bar(location_counts, x='Location', y='Pothole_Count', title='Pothole Count by Location', color='Pothole_Count')
    st.plotly_chart(fig_bar)

    fig_crack_bar = px.bar(location_counts, x='Location', y='Crack_Count', title='Crack Count by Location', color='Crack_Count')
    st.plotly_chart(fig_crack_bar)

    comparison_df = location_counts.melt(id_vars='Location', value_vars=['Pothole_Count', 'Crack_Count'], var_name='Type', value_name='Count')
    fig_comparison = px.bar(comparison_df, x='Location', y='Count', color='Type', barmode='group', title='Comparison of Potholes and Cracks by Location')
    st.plotly_chart(fig_comparison)

    most_potholes = location_counts.sort_values(by='Pothole_Count', ascending=False).head(5)
    st.write("Top 5 Neighborhoods with Most Potholes:")
    st.table(most_potholes[['Location', 'Pothole_Count']])

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

# Display the selected page
pages[selected_page]()
