<div align="center">

# 🧠 Artificial Neural Networks — UANL

![Kotlin](https://img.shields.io/badge/Kotlin-white?style=flat&logo=kotlin&logoColor=white&labelColor=00bcd4&color=0097a7)
![Android](https://img.shields.io/badge/Android-white?style=flat&logo=android&logoColor=white&labelColor=00bcd4&color=0097a7)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-white?style=flat&logo=tensorflow&logoColor=white&labelColor=00bcd4&color=0097a7)
![Figma](https://img.shields.io/badge/Figma-white?style=flat&logo=figma&logoColor=white&labelColor=00bcd4&color=0097a7)
![UANL](https://img.shields.io/badge/UANL-white?style=flat&logoColor=white&labelColor=00bcd4&color=0097a7)
![Status](https://img.shields.io/badge/Status-Complete-white?style=flat&logoColor=white&labelColor=00bcd4&color=0097a7)

> ⚠️ Code, comments and documentation are written in Spanish as part of my university coursework.

**Asistente Mágico de Escritura** is an Android app for children that uses a TensorFlow Lite neural network to recognize handwritten letters in real time, developed as the final project for the Artificial Neural Networks course at UANL.

</div>

---

## 📋 About

Young children often struggle with correctly forming letters, and traditional correction methods depend on constant teacher or parent supervision. This app offers an interactive alternative: kids draw uppercase letters on screen, and a trained neural network classifies their handwriting instantly, giving immediate visual feedback.

The model was trained using the [A-Z Handwritten Alphabets dataset](https://huggingface.co/datasets/pittawat/letter_recognition) and deployed as a `.tflite` file integrated directly into the Android app. The app was designed with a playful space-themed UI to keep young users engaged.

---

## ✨ Features

- Touch drawing canvas where children trace uppercase letters (A–Z)
- Real-time letter recognition using a TensorFlow Lite model running on-device
- Softmax probability output across all 26 classes
- Bitmap preprocessing pipeline — normalization, cropping, centering, and 28×28 rescaling
- "Nueva letra" button to randomize the target letter
- Fully offline — no data collected or sent externally

---

## 🛠️ Built With

![Kotlin](https://img.shields.io/badge/Kotlin-white?style=flat&logo=kotlin&logoColor=white&labelColor=00bcd4&color=0097a7)
![Android Studio](https://img.shields.io/badge/Android_Studio-white?style=flat&logo=androidstudio&logoColor=white&labelColor=00bcd4&color=0097a7)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-white?style=flat&logo=tensorflow&logoColor=white&labelColor=00bcd4&color=0097a7)
![Jetpack Compose](https://img.shields.io/badge/Jetpack_Compose-white?style=flat&logo=jetpackcompose&logoColor=white&labelColor=00bcd4&color=0097a7)
![Figma](https://img.shields.io/badge/Figma-white?style=flat&logo=figma&logoColor=white&labelColor=00bcd4&color=0097a7)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
