# Hi, I'm Aditya. 

I am a Software Developer and Data Scientist specializing in Deep Learning architectures, computer vision, and backend systems. I build production-grade AI applications and the data pipelines that power them.

### What I'm Building Right Now
* **Intelligent Perishable Logistics Engine:** Engineered an end-to-end supply chain routing system. Built a custom computer vision model using Transfer Learning (ResNet18) in PyTorch to predict biological decay margins, integrated with a dynamic Dijkstra's pathfinding algorithm for real-time Quick-Commerce routing. 

### Core Engineering Stack
* **Deep Learning & ML:** PyTorch, TensorFlow, Scikit-Learn
* **Languages:** Python, C++
* **Data Engineering & Vision:** OpenCV, Pandas, NumPy, Data Augmentation Pipelines
* **Deployment & Systems:** Streamlit, GitHub Actions, Linux/Bash

###  Highlighted Architecture
> **[Supply Chain AI & Vision Routing](Link to your Banana repo here)**
> * Mathematically optimized model training using Huber Loss and ReduceLROnPlateau scheduling, achieving a ~12-hour error margin on unseen test data.
> * Designed an active, heuristic-based routing layer that invalidates delivery paths based on real-time spoilage predictions.

###  Connect With Me
* [LinkedIn](linkedin.com/in/aditya-muchak-850180313)
* Reach out for discussions on ML architectures, optimization algorithms, or backend systems.

#  AI-Powered Perishable Logistics & Routing Engine

An end-to-end Machine Learning and Supply Chain Optimization system. This application uses a custom Deep Learning model to predict the exact biological shelf life of perishable goods (bananas) from an image, and dynamically routes them through a Quick-Commerce distribution network to prevent spoilage.

##  The Architecture

### 1. The Computer Vision Engine (PyTorch)
* **Architecture:** Transfer Learning via ResNet18 backbone.
* **Optimization:** Replaced the classification head with a custom continuous-value regression layer.
* **Training Protocol:** Implemented Huber Loss (Smooth L1) for outlier resistance and `ReduceLROnPlateau` scheduling to mathematically force model convergence.
* **Data Pipeline:** Built a dynamic data augmentation loader featuring randomized affine transformations to prevent environmental overfitting.

### 2. The Supply Chain Graph Engine (DSA)
* **Pathfinding:** Modified Dijkstra's Algorithm applied to a weighted directed graph of Central Warehouses, City Hubs, and Express Dark Stores.
* **Heuristic:** The algorithm actively compares the ML-predicted shelf life against route transit times, instantly blacklisting destination nodes where $T_{transit} > T_{shelf\_life}$.

### 3. The Deployment Layer (Streamlit)
* Live frontend caching AI weights onto the CPU for low-latency inference.
* Dynamic `Mermaid.js` integration to visually map valid vs. spoiled distribution networks in real time based on the payload image.
* the below are some outputs from the web
* <img width="1917" height="1000" alt="image" src="https://github.com/user-attachments/assets/925fb394-0946-4f68-9ac1-a9adad87df01" />

