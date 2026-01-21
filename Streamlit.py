import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ==========================================
# 1. CẤU HÌNH & CSS 
# ==========================================
st.set_page_config(page_title="ĐÁNH GIÁ MÀU SẮC CỦA NƯỚC NUÔI THỦY SẢN", page_icon="💧", layout="centered")

# CSS Custom 
st.markdown("""
    <style>
    [data-testid="stForm"] {
        background-color: #E0F7FA; /* Đổi sang màu xanh nước biển nhạt cho hợp chủ đề */
        padding: 20px !important;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMarkdown h4 {
        margin-top: -10px;
        font-weight: bold;
    }
    button[kind="primaryFormSubmit"] {
        background-color: #0277BD !important; /* Màu xanh đậm */
        color: white !important;
        border-radius: 10px !important;
        border: none;
        width: 100%;
        padding: 10px;
        font-size: 18px;
    }
    .stAlert {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HÀM LOAD MODEL RESNET50
# ==========================================
@st.cache_resource
def load_resnet50_model():
    """
    Hàm này khởi tạo kiến trúc ResNet50 và load trọng số đã train.
    """
    device = torch.device('cpu') # Web app chạy trên CPU cho đơn giản
    
    # 1. Khởi tạo kiến trúc model (phải giống hệt lúc train)
    # Vì lúc train bạn dùng weights='IMAGENET...', nên giờ gọi lại khung sườn đó
    model = models.resnet50(weights=None) 
    
    # 2. Thay đổi lớp cuối cùng (Fully Connected) cho 5 lớp đầu ra
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    
    # 3. Load trọng số từ file .pt
    # Lưu ý: map_location='cpu' để tránh lỗi nếu máy server không có GPU
    model_path = 'model_ResNet50.pt' 
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Chuyển sang chế độ đánh giá
        return model
    except FileNotFoundError:
        st.error(f"⚠️ LỖI: Không tìm thấy file '{model_path}'. Vui lòng copy file model vào cùng thư mục với file code này.")
        return None
    except Exception as e:
        st.error(f"⚠️ LỖI KHÁC: {e}")
        return None

# ==========================================
# 3. HÀM XỬ LÝ ẢNH (PRE-PROCESSING)
# ==========================================
def process_image(image):
    """
    Chuẩn hóa ảnh y hệt như lúc train
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Thêm chiều batch -> (1, 3, 224, 224)

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================

col_space_left, col_main, col_space_right = st.columns([0.1, 9.8, 0.1])

with col_main:
    c1, c2, c3 = st.columns([2, 1, 2]) 
    with c2:
        try:
            # Chỉnh width vừa phải (khoảng 120-150)
            st.image("logo-khtn.png", width=150)
        except:
            st.warning("Thiếu logo")

    #TÊN TRƯỜNG VÀ ĐỀ TÀI
    st.markdown("""
        <div style='text-align: center;'>
            <h3 style='color: black; margin-top: -15px; font-size: 22px;'>TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN, ĐHQG-HCM</h3>
            <h3 style='color: black; margin-top: -25px; font-size: 21.5px;'>KHOA VẬT LÝ – VẬT LÝ KỸ THUẬT</h3>
            <h3 style='color: black; margin-top: -25px; font-size: 21px;'>BỘ MÔN VẬT LÝ ĐIỆN TỬ</h3>
        </div>
    """, unsafe_allow_html=True)


    # Phần tên đề tài
    st.markdown("""
        <div style='text-align: center;'>
            <h3 style='color: #0288D1; margin-top: 20px; font-size: 25px;'>HỆ THỐNG PHÂN LOẠI MÀU NƯỚC AO NUÔI</h3>
            <h3 style='color: #0288D1; margin-top: -25px ; font-size: 25px;'>SỬ DỤNG MÔ HÌNH HỌC SÂU RESNET-50</h3>
            <p style='font-style: bold; color: red; margin-top: 10px; margin-bottom: -25px; font-size: 19px;'>Đồ Án Chuyên Đề - Hồ Thị Như Nguyệt</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

st.write("📸 **Hướng dẫn:** Vui lòng tải lên hình ảnh mẫu nước ao nuôi để hệ thống phân tích!!!.")

# --- Form Upload ---
with st.form("water_form"):
    uploaded_file = st.file_uploader("Chọn tệp hình ảnh (jpg, png, jpeg)...", type=["jpg", "png", "jpeg"])
    
    # Nút submit
    submit = st.form_submit_button("🔍 PHÂN TÍCH KẾT QUẢ")

# ==========================================
# 5. XỬ LÝ DỰ ĐOÁN
# ==========================================
if submit:
    if uploaded_file is None:
        st.warning("Vui lòng chọn ảnh trước khi bấm phân tích!")
    else:
        # Hiển thị ảnh vừa upload
        image = Image.open(uploaded_file).convert('RGB')
        
        # Chia cột: Bên trái ảnh, Bên phải kết quả
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.image(image, caption="Ảnh nước ao nuôi", use_container_width=True)
        
        with c2:
            with st.spinner('Đang xử lý qua mạng ResNet50...'):
                # Load model
                model = load_resnet50_model()
                
                if model is not None:
                    # Xử lý ảnh và dự đoán
                    img_tensor = process_image(image)
                    
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1) # Tính %
                        confidence, preds = torch.max(probs, 1)
                        
                    # Mapping kết quả (Label 1 -> 5)
                    # Chú ý: Thứ tự này phải KHỚP với thứ tự class_names lúc train
                    class_names = [
                        "Label 1 - Màu Vàng", 
                        "Label 2 - Màu Vàng Nhạt", 
                        "Label 3 - Màu Vàng Xanh", 
                        "Label 4 - Màu Xanh Biển Nhạt", 
                        "Label 5 - Màu Xanh Biển"
                    ]
                    
                    # Lấy nhãn và độ tin cậy
                    pred_label = class_names[preds.item()]
                    conf_score = confidence.item() * 100
                    
                    # Hiển thị kết quả
                    st.success("✅ PHÂN TÍCH HOÀN TẤT")
                    st.markdown(f"### Kết quả: **{pred_label}**")
                    st.metric(label="Độ tin cậy của mô hình", value=f"{conf_score:.2f}%")
                    