import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from streamlit_chat import message
import openai
import random
import numpy as np
from PIL import Image
import sqlite3  

st.set_page_config(page_title="Pothole Detection", page_icon="🚧", layout="wide")

openai.api_key = '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' 

# Load YOLO model
model = YOLO("best (3).pt")

faq_df = pd.read_csv('pothole_detection_faq.csv')  
faq_df.columns = faq_df.columns.str.strip() 

pothole_data = pd.read_csv('updated_pothole_data.csv')
pothole_data.columns = pothole_data.columns.str.strip()  

def apply_custom_css():
    st.markdown("""
        <style>
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
        </style>
    """, unsafe_allow_html=True)


#def get_drilling_data():
    #conn = sqlite3.connect('drilling_data.db')  
    #cursor = conn.cursor()
    
    
    #cursor.execute('SELECT latitude, longitude FROM drilling_records')
    #data = cursor.fetchall()

    #conn.close()  
    #return data

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
    image = image.convert("RGB")
    img = np.array(image)
    results = model(img)
    
    pothole_count = len(results[0].boxes)  
    result_image = results[0].plot()  

    return result_image, pothole_count

def get_response(user_input, pothole_count):
    report = f"I detected {pothole_count} pothole(s) in the media you provided."
    
    if pothole_count == 0:
        recommendations = "No action needed. The road appears to be in good condition."
    elif 1 <= pothole_count <= 3:
        recommendations = "Consider patching the potholes to improve road safety."
    else:
        recommendations = "Immediate maintenance is recommended to address the multiple potholes."

    context = f"{report}\n{recommendations}\n\n"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about pothole detection."},
            {"role": "user", "content": context + user_input}
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

            pothole_severity = classify_surface_condition(pothole_count)

            pothole_data = {
                "Severity": [pothole_severity],
                
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
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        result_image, pothole_count = detect_potholes_image(image)

        st.image(result_image, caption="Detected Potholes", use_column_width=True)
        st.success(f"Detected {pothole_count} potholes in the image.")

        
        chatbot(pothole_count)

def dashboard_page():
    st.title("Pothole Detection Dashboard")
    st.write("This page shows visualizations of detected potholes and cracks.")

    m = folium.Map(location=[24.774265, 46.738586], zoom_start=13, tiles="CartoDB dark_matter")

    for _, row in pothole_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Pothole ID: {row['Pothole_ID']}<br>Location: {row['Location']}<br>Date Detected: {row['Date_Detected']}<br>Pothole Count: {row['Pothole_Count']}<br>Crack Count: {row['Crack_Count']}",
            icon=folium.Icon(color='red')
        ).add_to(m)

    st_folium(m, width=900, height=800)

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


import plotly.express as px
from datetime import datetime
import pandas as pd
import folium
from folium.plugins import HeatMap

def heatmap_page():
    st.title("Pothole Heatmap with Selection")
    st.write("This page shows a table of potholes and allows you to select one to see its location and set its status.")

    # تحميل الملف الجديد باستخدام الفاصلة المنقوطة كفاصل
    new_file_path = 'potholes road detection _location_data.csv'  # تأكد من وضع المسار الصحيح للملف
    new_pothole_data = pd.read_csv(new_file_path, delimiter=';')

    # عرض أسماء الأعمدة الموجودة في الملف للتحقق منها
    st.write("Columns in the file:", new_pothole_data.columns)

    # التحقق مما إذا كان العمود 'Location' موجود
    if 'Location' in new_pothole_data.columns:
        # تقسيم عمود 'Location' إلى أعمدة 'Longitude' و 'Latitude'
        new_pothole_data[['Latitude', 'Longitude']] = new_pothole_data['Location'].str.split(',', expand=True)

        # تحويل القيم في أعمدة Latitude و Longitude إلى قيم رقمية
        new_pothole_data['Latitude'] = pd.to_numeric(new_pothole_data['Latitude'], errors='coerce')
        new_pothole_data['Longitude'] = pd.to_numeric(new_pothole_data['Longitude'], errors='coerce')

        # إضافة معرف فريد لكل حفرة بناءً على الترتيب
        new_pothole_data['Pothole_ID'] = new_pothole_data.index + 1

        # إضافة إحداثيات جديدة
        new_entry = {
            'Pothole_ID': new_pothole_data['Pothole_ID'].max() + 1,  # توليد معرف جديد بناءً على آخر معرف موجود
            'Time': datetime.today().strftime('%Y-%m-%d'),
            'Latitude': 24.85380,
            'Longitude': 46.71320,
            'تمت المعالجة': False,
            'تحت المعالجة': False,
            'لم تتم المعالجة': True
        }

        # إضافة الحفرة الجديدة إلى البيانات
        new_pothole_data = new_pothole_data.append(new_entry, ignore_index=True)

        # فلترة البيانات لإزالة الصفوف التي تحتوي على قيم مفقودة
        valid_data = new_pothole_data.dropna(subset=['Latitude', 'Longitude'])

        if not valid_data.empty:
            # اختيار الحفرة من القائمة
            selected_pothole = st.selectbox("Select a pothole to update its status:", valid_data['Pothole_ID'])

            # عرض البيانات الخاصة بالحفرة المختارة
            selected_data = valid_data[valid_data['Pothole_ID'] == selected_pothole]
            st.write("Selected Pothole Details:")
            st.table(selected_data[['Pothole_ID', 'Time', 'Latitude', 'Longitude']])

            # تحديث الحالة بناءً على اختيارات المستخدم
            st.write("Update Pothole Status:")
            valid_data.loc[valid_data['Pothole_ID'] == selected_pothole, 'تمت المعالجة'] = st.checkbox("تمت المعالجة", key=f"fixed_{selected_pothole}")
            valid_data.loc[valid_data['Pothole_ID'] == selected_pothole, 'تحت المعالجة'] = st.checkbox("تحت المعالجة", key=f"in_progress_{selected_pothole}")
            valid_data.loc[valid_data['Pothole_ID'] == selected_pothole, 'لم تتم المعالجة'] = not (
                valid_data.loc[valid_data['Pothole_ID'] == selected_pothole, 'تمت المعالجة'].values[0] or
                valid_data.loc[valid_data['Pothole_ID'] == selected_pothole, 'تحت المعالجة'].values[0]
            )

            # عرض الخريطة مع الموقع المحدد للحفرة
            m = folium.Map(location=[24.774265, 46.738586], zoom_start=13, tiles="CartoDB dark_matter")
            for index, row in selected_data.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"Pothole ID: {row['Pothole_ID']}<br>Time: {row['Time']}<br>Status: {row['تمت المعالجة']}"
                ).add_to(m)

            # عرض الخريطة في Streamlit
            st_folium(m, width=900, height=800)

            # عرض إحصائيات التفاعل بناءً على حالة المعالجة
            st.subheader("Pothole Processing Statistics")

            # حساب عدد الحفر بناءً على اختيار المستخدم
            status_counts = {
                'تمت المعالجة': valid_data['تمت المعالجة'].sum(),
                'تحت المعالجة': valid_data['تحت المعالجة'].sum(),
                'لم تتم المعالجة': valid_data['لم تتم المعالجة'].sum()
            }

            # رسم الدائرة التفاعلية
            fig = px.pie(
                names=status_counts.keys(), 
                values=status_counts.values(), 
                title='Pothole Processing Status'
            )
            st.plotly_chart(fig)

            # *** إضافة الخريطة الحرارية ***
            st.subheader("Heatmap of Pothole Locations")

            # إنشاء خريطة حرارية باستخدام المواقع الصالحة
            heatmap_data = valid_data[['Latitude', 'Longitude']].dropna().values.tolist()

            # إنشاء خريطة جديدة للحرارة
            heatmap_map = folium.Map(location=[24.774265, 46.738586], zoom_start=13, tiles="CartoDB dark_matter")
            HeatMap(heatmap_data).add_to(heatmap_map)

            # عرض الخريطة الحرارية في Streamlit
            st_folium(heatmap_map, width=900, height=800)

        else:
            st.error("No valid data available in the file.")
    else:
        st.error("Column 'Location' not found in the file.")










def neighborhoods_page():
    st.title("Neighborhoods in Riyadh")
    st.write("Click on a neighborhood to see pothole details.")

    neighborhoods = {
        "Al-Narjis": {"location": [24.832592631493867, 46.682551601963375], "potholes": 660, "severity": "Severe"},
        "Al-Munsiyah": {"location": [24.839311029020966, 46.77422508429597], "potholes": 70, "severity": "low Risk"},
        "Qurtubah": {"location": [24.812424274752065, 46.75313865668322], "potholes": 100, "severity": "Severe"}
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

def chatbot(pothole_count):
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)  
    st.markdown('<div class="chat-header">Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

    
    for i in range(len(st.session_state['generated'])):
        st.markdown(f'<div class="chat-message user-message">{st.session_state["past"][i]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message bot-message">{st.session_state["generated"][i]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  

    
    user_input = st.text_input("اسألني عن كشف الحفر", key="user_input")
    
    if st.button("إرسال"):
        if user_input:
            output = get_response(user_input, pothole_count)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    st.markdown('</div>', unsafe_allow_html=True)  


    st.markdown('</div>', unsafe_allow_html=True)

# Navigation
pages = {
    "Main Page": main_page,
    "Dashboard": dashboard_page,
    "Heatmap": heatmap_page,
    "Neighborhoods": neighborhoods_page,
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", pages.keys())

apply_custom_css()

# Display the selected page
pages[selected_page]()
