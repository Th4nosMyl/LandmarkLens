"""
app.py

Μια εφαρμογή αναγνώρισης μνημείων (landmarks) βασισμένη σε ένα fine-tuned CLIP μοντέλο.
Χρησιμοποιεί το Streamlit για το UI και προαιρετικά αντλεί πληροφορίες από τη Wikipedia.
"""

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import json
import wikipedia

# -----------------------------------------
# 1) Εισαγωγή της κλάσης από το Python αρχείο
# -----------------------------------------
# Στο φάκελο "models", βρίσκεται το "clip_classifier.py",
# όπου ορίζεται η κλάση CLIPWithClassifier (nn.Module).
from models.clip_classifier import CLIPWithClassifier


# -----------------------------------------
# 2) Συνάρτηση φόρτωσης του μοντέλου
# -----------------------------------------
@st.cache_resource
def load_model():
    """
    Φορτώνει το CLIP Image Encoder + το Linear Classifier (CLIPWithClassifier) 
    και τα βάρη από το .pth checkpoint. 

    Returns:
        model (nn.Module): Το CLIPWithClassifier σε κατάσταση .eval().
        device (str): Συσκευή (cuda ή cpu) για χρήση στο inference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import clip  # pip install git+https://github.com/openai/CLIP.git
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model = clip_model.float()

    # Ο αριθμός των κλάσεων πρέπει να συμπίπτει με αυτόν που χρησιμοποιήθηκε στο training.
    num_classes = 1434  # Παράδειγμα

    model = CLIPWithClassifier(
        clip_model=clip_model,
        num_classes=num_classes,
        dropout_rate=0.1,
        freeze_clip=False
    ).to(device)

    checkpoint_path = "models/BestClipModel.pth"  # Προσαρμόστε το path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


# -----------------------------------------
# 3) Συνάρτηση φόρτωσης του label_mapping
# -----------------------------------------
@st.cache_resource
def load_label_mapping():
    """
    Φορτώνει το JSON που χαρτογραφεί (encoded_label -> friendly_name).
    Δημιουργείται στο Notebook, συνήθως από: 
    (encoded_label -> landmark_id) + (landmark_id -> categoryURL) -> friendly_name.

    Returns:
        label_map (dict[int,str]): Λεξικό όπου το κλειδί είναι το int encoded_label
                                   και η τιμή είναι το ανθρώπινο όνομα του landmark.
    """
    mapping_json_path = "data/label_mapping.json"
    with open(mapping_json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    label_map = {int(k): v for k, v in mapping.items()}
    return label_map


# -----------------------------------------
# 4) Συνάρτηση για λήψη περίληψης από Wikipedia (προαιρετική)
# -----------------------------------------
def get_wiki_summary(title: str, lang="en", sentences=2):
    """
    Παίρνει σύντομη περίληψη από τη Wikipedia για ένα δεδομένο "title".

    Args:
        title (str): Το λήμμα για αναζήτηση.
        lang (str): Γλώσσα (default="en").
        sentences (int): Πόσες προτάσεις στην περίληψη.

    Returns:
        (str or None): 
            - Κείμενο περίληψης αν βρεθεί,
            - None αν υπάρχει disambiguation ή σφάλμα σελίδας.
    """
    try:
        wikipedia.set_lang(lang)
        summary = wikipedia.summary(title, sentences=sentences)
        return summary
    except wikipedia.DisambiguationError:
        return None
    except wikipedia.PageError:
        return None
    except Exception:
        return None


# -----------------------------------------
# 5) Συναρτήσεις για Preprocessing & Predict
# -----------------------------------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Μετασχηματισμοί που χρησιμοποιήθηκαν στο training:
      - Resize (224,224)
      - ToTensor
      - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

    Args:
        image (PIL.Image): Εικόνα σε μορφή PIL.

    Returns:
        torch.Tensor: Σχήμα [1,3,224,224].
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def predict(image_tensor: torch.Tensor, model: torch.nn.Module, device: str):
    """
    Κάνει forward pass στο μοντέλο (χωρίς gradient) και επιστρέφει top-5 κλάσεις/πιθανότητες.

    Args:
        image_tensor (torch.Tensor): shape [1,3,224,224].
        model (nn.Module): CLIPWithClassifier σε eval().
        device (str): 'cuda' ή 'cpu'.

    Returns:
        (np.array, np.array):
            top_probs: Οι top-5 πιθανότητες
            top_classes: Οι top-5 αντίστοιχες κλάσεις
    """
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        top_probs, top_classes = probs.topk(5, dim=1)
        return top_probs.cpu().numpy()[0], top_classes.cpu().numpy()[0]


# -----------------------------------------
# 6) Συνάρτηση: decode_predictions (με Wiki)
# -----------------------------------------
def decode_predictions(top_probs, top_classes, label_map, wiki_lang="en"):
    """
    Μετατρέπει τις top-5 προβλέψεις (probabilities & class indices)
    σε φιλικό κείμενο. Καλεί προαιρετικά wikipedia.summary.

    Args:
        top_probs (np.array): π.χ. [0.45,0.22,0.15,0.10,0.08]
        top_classes (np.array): π.χ. [23,89,17,104,356]
        label_map (dict[int,str]): mapping int->str (0->"Eiffel Tower" κλπ.)
        wiki_lang (str): Γλώσσα Wikipedia. Default="en"

    Returns:
        list[str]: Λίστα από strings για εμφάνιση.
    """
    results = []
    for prob, cls_idx in zip(top_probs, top_classes):
        landmark_name = label_map.get(cls_idx, f"Landmark_{cls_idx}")
        summary = get_wiki_summary(landmark_name, lang=wiki_lang, sentences=2)
        if summary:
            text = f"**{landmark_name}** - {prob*100:.2f}%\n\n{summary}"
        else:
            text = f"**{landmark_name}** - {prob*100:.2f}%\n\n(No summary found)"
        results.append(text)
    return results


# -----------------------------------------
# 7) Streamlit UI - Main λογική
# -----------------------------------------
def main():
    """
    Βασική συνάρτηση του Streamlit app.
    Φορτώνει το μοντέλο & το label mapping, προσφέρει Upload/Camera,
    και εμφανίζει τις top-5 προβλέψεις CLIP.
    """
    st.title("Landmark Recognition App \U0001F5BC")
    st.markdown("""
    Καλωσήρθατε στην εφαρμογή **Landmark Recognition**!  
    Χρησιμοποιεί ένα προσαρμοσμένο (fine-tuned) CLIP μοντέλο για να εντοπίσει
    μνημεία/τοποθεσίες σε οποιαδήποτε εικόνα, εμφανίζοντας και σύντομη περίληψη
    από τη Wikipedia (προαιρετικά).
    """)

    st.sidebar.title("Πληροφορίες & Ρυθμίσεις")
    st.sidebar.markdown("""
    **Πηγή Δεδομένων**: Google Landmarks Dataset v2  
    **Μοντέλο**: CLIP (ViT-B/32) + Linear Classifier  
    **Χρήση**: Επιλέξτε εικόνα (Upload/Camera).  
    """)

    # Φορτώνουμε το μοντέλο & το label mapping
    model, device = load_model()
    label_map = load_label_mapping()

    # Επιλογή πηγής εικόνας
    st.write("### 1) Επιλογή Πηγής Εικόνας")
    option = st.radio(
        "Πώς θέλετε να εισάγετε την εικόνα;", 
        ("Upload", "Camera"), 
        index=0
    )

    image = None
    if option == "Upload":
        uploaded_file = st.file_uploader("Ανεβάστε αρχείο (.jpg ή .png)", type=["jpg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception:
                st.error("Σφάλμα κατά το άνοιγμα της εικόνας. Δοκιμάστε ξανά.")
    else:
        camera_file = st.camera_input("Τραβήξτε φωτογραφία από την κάμερα")
        if camera_file is not None:
            try:
                image = Image.open(camera_file).convert("RGB")
            except Exception:
                st.error("Σφάλμα κατά το άνοιγμα της εικόνας από την κάμερα.")

    if image is not None:
        st.write("### 2) Προεπισκόπηση Εικόνας")
        st.image(image, caption="Επιλεγμένη Εικόνα", use_container_width=True)

        st.write("### 3) Αποτελέσματα Αναγνώρισης")
        with st.spinner("Παρακαλώ περιμένετε… Αναγνωρίζουμε το μνημείο!"):
            # Preprocess & Predict
            image_tensor = preprocess_image(image)
            top_probs, top_classes = predict(image_tensor, model, device)
            # Decode με Wiki info
            results = decode_predictions(top_probs, top_classes, label_map, wiki_lang="en")

        st.success("Η αναγνώριση ολοκληρώθηκε με επιτυχία!")
        st.subheader("Top-5 Προβλέψεις:")
        for idx, line in enumerate(results, start=1):
            st.markdown(f"**#{idx}**\n{line}")
            st.markdown("---")
    else:
        st.info("Ανεβάστε ή τραβήξτε μια φωτογραφία για να ξεκινήσει η αναγνώριση.")


# ----------------------------
# 8) Footer
# ----------------------------
def custom_footer():
    """
    Εμφανίζει ένα custom footer στο κάτω μέρος της σελίδας,
    με emojis και links.
    """
    footer_html = """
    <hr/>
    <div style="text-align: center; padding: 10px;">
        <p>🔧 <strong>Προγραμματιστής:</strong> 
            <a href="mailto:Th4nosMylonas@gmail.com" target="_blank">Θανάσης Μυλωνάς</a> &nbsp;|&nbsp;
            🌐 <strong>GitHub:</strong> 
            <a href="https://github.com/Th4nosMyl" target="_blank">Th4nosMyl</a>
        </p>
        <p>© 2024 Landmark Lens App &nbsp;|&nbsp; 🏛️ Κατασκευή: 
           <em>CLIP Fine-Tuning</em></p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


# Σημείο Εκκίνησης του Streamlit App
if __name__ == "__main__":
    main()
    custom_footer()  # Εμφανίζει το footer στο τέλος
