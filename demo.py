import gradio as gr
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger ton modèle TensorFlow entraîné
model = tf.keras.models.load_model('modele.keras')

# Charger le tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Paramètres pour la tokenisation
vocab_size = 10000  # taille du vocabulaire
max_length = 100    # longueur maximale des séquences
embedding_dim = 100 # dimension des embeddings

# Fonction de prédiction
def predire_spam(texte_email):
    # Tokenisation du texte
    input_data = tokenizer.texts_to_sequences([texte_email])  # Convertir le texte en une séquence de nombres

    # Padding de la séquence pour la rendre de la même longueur
    input_data = pad_sequences(input_data, maxlen=max_length, padding='post')  # Applique le padding

    # Prédiction avec le modèle
    prediction = model.predict(input_data)

    # Si ton modèle retourne une probabilité, tu peux définir un seuil (par exemple 0.5)
    if prediction[0] > 0.5:
        return f"Spam et la probabilité de spam est : {prediction[0][0]:.4f}"
    else:
        return f"Non Spam et la probabilité de spam est : {prediction[0][0]:.4f}"

# Créer l'interface Gradio
interface = gr.Interface(fn=predire_spam,
                         inputs=gr.Textbox(label="Entrez l'email", placeholder="Écrivez un email ici...", lines=5),
                         outputs="text",  # Affichage texte pour le résultat
                         title="Détecteur de Spam",
                         description="Entrez un e-mail pour détecter s'il est un spam ou non.")

# Lancer l'application Gradio
interface.launch(share=True)
