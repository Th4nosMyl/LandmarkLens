# Landmark Recognition Project

> Μια **προσαρμοσμένη εφαρμογή** αναγνώρισης μνημείων (landmarks) βασισμένη στο CLIP μοντέλο της OpenAI, σε συνδυασμό με ένα Linear Classifier.  

---

## Περιεχόμενα

1. [Επισκόπηση Project](#επισκόπηση-project)  
2. [Δεδομένα & Φιλτράρισμα](#δεδομένα--φιλτράρισμα)  
3. [Προσέγγιση & Εκπαίδευση](#προσέγγιση--εκπαίδευση)  
4. [Δημιουργία Label Mapping](#δημιουργία-label-mapping)  
5. [Οδηγίες Χρήσης (Local ή Cloud)](#οδηγίες-χρήσης-local-ή-cloud)  
6. [Δομή Αρχείων](#δομή-αρχείων)  
7. [Προϋποθέσεις & Εγκατάσταση](#προϋποθέσεις--εγκατάσταση)  
8. [Συχνές Ερωτήσεις](#συχνές-ερωτήσεις)  
9. [Credits & Επικοινωνία](#credits--επικοινωνία)

---

## Επισκόπηση Project

Σε αυτό το project, στοχεύουμε στην **αναγνώριση μνημείων** (landmarks) από εικόνες. Χρησιμοποιούμε:

- [**CLIP**](https://github.com/openai/CLIP) (ViT-B/32) ως backbone για την εξαγωγή χαρακτηριστικών (image encoder).  
- Έναν **Linear Classifier** πάνω από τα χαρακτηριστικά του CLIP, εκπαιδευμένο σε συγκεκριμένες τοποθεσίες/μνημεία.  
- **Streamlit** για το UI (web app), όπου ο χρήστης μπορεί να ανεβάσει φωτογραφίες ή να βγάλει φωτογραφία μέσω κάμερας.  
- Προαιρετικά, καλούμε το **Wikipedia API** για να εμφανίσουμε μια σύντομη περιγραφή του μνημείου.

**Βασική ιδέα**: Το CLIP είναι ήδη προεκπαιδευμένο σε πολλά δεδομένα (image-text pairs), οπότε απαιτείται λιγότερο data για fine-tuning. Εμείς προσαρμόζουμε το μοντέλο σε συγκεκριμένο σύνολο landmarks (Google Landmarks Dataset v2), προσθέτοντας μόνο ένα Linear layer στο τέλος.

---

## Δεδομένα & Φιλτράρισμα

Χρησιμοποιούμε το [**Google Landmarks Dataset v2**](https://github.com/visipedia/google-landmark):

1. **Φιλτράρισμα**:  
   - Αφαιρούμε όσες εικόνες δεν υπάρχουν τοπικά ή είναι κατεστραμμένες,  
   - Αποκλείουμε όσες κλάσεις (landmark_id) έχουν πολύ λίγα δείγματα (π.χ. <5 ή <10 εικόνες).  

2. **final_filtered_train.csv**:  
   - Προκύπτει μετά από το φιλτράρισμα, περιλαμβάνοντας μόνο κλάσεις με επαρκή αριθμό εικόνων.

3. **Encoding**:  
   - Δημιουργήσαμε στήλη `encoded_label` \([0..N-1]\) που αντιστοιχεί σε κάθε `landmark_id`. Εδώ αποθηκεύεται η νέα ετικέτα που καταλαβαίνει το ταξινομητικό μοντέλο (PyTorch).

> **Σημαντικό**: Όλα αυτά τα βήματα γίνονται σε Notebook, ώστε να προετοιμάσουμε σωστά το dataset για εκπαίδευση.

---

## Προσέγγιση & Εκπαίδευση

1. **Fine-tuning CLIP**:  
   - Φορτώνουμε το προεκπαιδευμένο CLIP (ViT-B/32).  
   - Προσθέτουμε ένα `Linear` layer (output_dim = `clip_model.visual.output_dim`, input_dim = `num_classes`).  

2. **Training** (PyTorch):  
   - Optimizer: Adam  
   - Loss: CrossEntropyLoss  
   - LR Scheduler: [ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)  
   - EarlyStopping (με `patience`) για να αποφύγουμε overfitting σε πολλές εποχές.

3. **Checkpoint**:  
   - Αποθηκεύουμε το καλύτερο μοντέλο (lowest val_loss) ως `BestClipModel.pth`.

4. **Validation**:  
   - Ελέγχουμε την απόδοση σε `val_loader`. Βλέπουμε αρκετά υψηλά ποσοστά (>95%).  
   - Στη συνέχεια, το τελικό μοντέλο φορτώνεται στο `app.py` για το inference.

---

## Δημιουργία Label Mapping

1. **Encoded Label**: Στο CSV (`final_filtered_train_encoded.csv`) έχουμε `encoded_label` για κάθε δείγμα.  
2. **landmark_id → category URL**: Από το `train_label_to_category.csv`.  
3. **Φιλικό όνομα**: Κόβουμε το `"Category:Eiffel_Tower"` σε `"Eiffel Tower"`.  
4. **label_mapping.json**: Φτιάχνουμε ένα dict `encoded_label -> friendly_name` και το αποθηκεύουμε.  

**Στο `app.py`**: Όταν το μοντέλο προβλέπει μια κλάση #42, εμείς διαβάζουμε `label_mapping[42] = "Eiffel Tower"`, άρα εμφανίζουμε κανονικά ονόματα.

---

## Οδηγίες Χρήσης 

1. **Clone** ή **Download** το repository:

   ```bash
   git clone https://github.com/Th4nosMyl/LandmarkLens.git
   cd LandmarkLens
   ```

2. Βεβαιωθείτε πως το `BestClipModel.pth` υπάρχει στον φάκελο `models/`, και το `label_mapping.json` στον φάκελο `data/`.

3. Εγκαταστήστε τις εξαρτήσεις:

   ```bash
   pip install -r requirements.txt
   ```

4. Τρέξτε:

   ```bash
   streamlit run app.py
   ```

5. Ανοίξτε το browser (συνήθως http://localhost:8501) για να δείτε την εφαρμογή.

- Επιλέξτε **Upload** ή **Camera**.  
- Δώστε μια εικόνα ή βγάλτε φωτογραφία.  
- Δείτε τις top-5 προβλέψεις, τις πιθανότητες και (προαιρετικά) μια σύντομη wiki summary.

---

## Δομή Αρχείων

### Παράδειγμα αρχιτεκτονικής φακέλων:

```plaintext
LandmarkLens/
├── app.py                 # Το Streamlit app
├── requirements.txt       # Εξαρτήσεις
├── models/
│   ├── clip_classifier.py # Κλάση CLIPWithClassifier (PyTorch)
│   └── BestClipModel.pth  # Βαρίδια (checkpoint)
├── data/
│   └── label_mapping.json # Mapping: (encoded_label -> friendly_name)
├── README.md              # Το παρόν αρχείο
└── ... (άλλα notebooks ή scripts)
```

---

### Προϋποθέσεις & Εγκατάσταση

- **Python 3.8+** (συνίσταται).
- **PyTorch** (με CPU ή CUDA, αναλόγως δυνατοτήτων).
- **Βιβλιοθήκες**: streamlit, wikipedia, Pillow, torchvision, κ.λπ. (δείτε `requirements.txt`).

---

## Βασικά βήματα:

```bash
pip install -r requirements.txt
```

Εάν επιθυμείτε GPU επιτάχυνση, εγκαταστήστε κατάλληλη έκδοση PyTorch με CUDA:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
```

---

## Συχνές Ερωτήσεις

- **Γιατί το validation accuracy είναι πολύ υψηλό, αλλά σε νέες εικόνες οι προβλέψεις πέφτουν;**  
  Πιθανότατα υπάρχει **data leakage** ή εύκολο validation split. Χρήσιμο να έχετε ανεξάρτητο test set ή να χρησιμοποιείτε cross-validation.

- **Τι γίνεται αν δεν βρει σελίδα Wikipedia;**  
  Η συνάρτηση `get_wiki_summary` επιστρέφει `(No summary found)`. Μπορείτε να αφαιρέσετε το Wiki εάν δεν θέλετε επιπλέον πληροφορίες.

- **Μπορώ να προσθέσω έξτρα landmarks;**  
  Ναι, προσθέστε εικόνες, επανεκπαιδεύστε το μοντέλο και ενημερώστε το `label_mapping.json`.

- **Έχω σφάλμα στο app.py “File not found: label_mapping.json”!**  
  Βεβαιωθείτε ότι το `label_mapping.json` βρίσκεται στον φάκελο `data/` ή αλλάξτε το path στο `app.py`.

---

## Credits & Επικοινωνία

- **Dataset**: Google Landmarks Dataset v2.  
- **Μοντέλο**: OpenAI CLIP.  
- **UI**: Streamlit.  

### Developer: Th4nosMyl

- **Email**: Th4nosMylonas@gmail.com  
- **GitHub**: [github.com/Th4nosMyl](https://github.com/Th4nosMyl)  

---

<p align="center">Made with :heart: for Educational Purposes | LandmarkLens</p>
