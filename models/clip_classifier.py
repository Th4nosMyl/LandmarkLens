"""
clip_classifier.py

Περιέχει τον ορισμό της κλάσης CLIPWithClassifier, η οποία επεκτείνει ένα
προεκπαιδευμένο CLIP μοντέλο (image encoder) προσθέτοντας ένα γραμμικό ταξινομητή
(linear layer) για πολυταξινόμηση εικόνων.

Παράδειγμα Χρήσης:
------------------
from clip_classifier import CLIPWithClassifier

# Αφού έχεις ήδη φορτώσει το clip_model (π.χ. clip.load("ViT-B/32")),
# μπορείς να δημιουργήσεις:
model = CLIPWithClassifier(
    clip_model=clip_model,
    num_classes=1434,
    dropout_rate=0.1,
    freeze_clip=False
)
# ...προαιρετικά, κάνεις training, ή φορτώνεις βάρη από checkpoint...
"""

import torch
import torch.nn as nn
import clip  # Βεβαιώσου ότι έχεις εγκατεστημένο το clip με: pip install git+https://github.com/openai/CLIP.git

class CLIPWithClassifier(nn.Module):
    """
    Κλάση που επεκτείνει τη λογική ενός προεκπαιδευμένου CLIP μοντέλου (image part)
    προσθέτοντας ένα Linear Layer ταξινομητή στο τέλος. Έτσι, μπορούμε να
    εκπαιδεύσουμε/χρησιμοποιήσουμε το μοντέλο για "κανονική" πολυταξινόμηση εικόνων.

    Args:
        clip_model (nn.Module): Το προεκπαιδευμένο μοντέλο CLIP (π.χ. από clip.load(...)).
        num_classes (int): Αριθμός κλάσεων για την ταξινόμηση.
        dropout_rate (float): Ρυθμός dropout πριν το τελικό Linear Layer.
        freeze_clip (bool): Αν είναι True, παγώνουμε (requires_grad=False) όλα τα βάρη
                           του CLIP encoder, ώστε να εκπαιδεύεται μόνο το Linear Layer.
    """
    def __init__(self, clip_model, num_classes, dropout_rate=0.1, freeze_clip=False):
        super(CLIPWithClassifier, self).__init__()
        
        # Το βασικό image encoder από το CLIP.
        # Περιλαμβάνει layers για την εξαγωγή χαρακτηριστικών (encode_image).
        self.clip_model = clip_model
        
        # Προσθέτουμε dropout πριν το τελικό layer για τακτική.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Ο τελικός ταξινομητής (Linear Layer).
        # Εδώ, χρησιμοποιούμε self.clip_model.visual.output_dim για το input size.
        self.classifier = nn.Linear(
            self.clip_model.visual.output_dim,
            num_classes
        )

        # Αν θέλουμε να παγώσουμε το CLIP, ορίζουμε requires_grad=False στα βάρη του.
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass στο μοντέλο. Αρχικά αντλούμε χαρακτηριστικά από το CLIP encoder,
        μετά εφαρμόζουμε dropout, και τέλος περνάμε τα χαρακτηριστικά
        από το Linear Layer για να πάρουμε logits για κάθε κλάση.

        Args:
            images (torch.Tensor): Batch από εικόνες, τυπικά shape [B, 3, H, W].
                                   Συνήθως H,W = 224, και έχουν ήδη κανονικοποιηθεί.

        Returns:
            logits (torch.Tensor): Ένα tensor με shape [B, num_classes],
                                   που περιέχει τα ακατέργαστα logits (πριν το softmax).
        """
        # Χρησιμοποιούμε την ενσωματωμένη συνάρτηση encode_image του CLIP.
        # Επιστρέφει ένα embedding με μέγεθος self.clip_model.visual.output_dim.
        image_features = self.clip_model.encode_image(images)

        # Εφαρμόζουμε dropout.
        x = self.dropout(image_features)

        # Τελικά logits για ταξινόμηση.
        logits = self.classifier(x)
        return logits
