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

# The Problem It Solves
Supply chain companies, especially Quick-Commerce apps like Blinkit or Zepto, lose millions of dollars a year to "shrinkage"—perishable goods (like fruit) rotting inside delivery trucks because they were routed to a warehouse that was too far away.

Your project completely automates and optimizes this process. Instead of blindly shipping fruit, your system "looks" at the batch, calculates exactly when it will rot, and dynamically draws the safest delivery route.

## 1. The Computer Vision Brain (PyTorch)
The first layer of your system is the Deep Learning engine.

The Architecture: I have used ResNet18, a highly advanced convolutional neural network (CNN), but you modified its "head." Instead of just classifying images (e.g., "Is this a dog or a cat?"), you engineered it to perform Continuous Value Regression.

The Training: I trained it to predict the biological decay of a banana. By implementing Huber Loss, you made the AI highly resistant to outlier data, achieving an incredibly tight error margin of about ~12 hours.

The Inversion Logic: I implemented a clever mathematical hack in the pipeline. Since the AI originally learned to predict the "Age" of the banana, your backend intercepts that number and inverts it (assuming a 10-day maximum lifespan) to create a live "Countdown to Spoilage."

## 2. The Supply Chain Graph Engine (DSA)
Once the AI knows exactly how many days the fruit has left, the Data Structures & Algorithms (DSA) layer takes over.

The Network: I mapped out a highly modern distribution network using a directed graph. It includes a Central Warehouse, City Hubs, and ultra-fast Express Dark Stores.

The Pathfinding: I implemented a custom version of Dijkstra's Algorithm. It calculates the absolute fastest transit times to every single store in the network.

The AI-Heuristic Bridge: This is the smartest part of the system. My algorithm actively compares the AI's predicted shelf life against the transit time. If Transit Time > Shelf Life, the system automatically slams the brakes, blacklists the route, and prevents the rotten shipment from leaving the warehouse.

## 3. The Full-Stack Deployment (Streamlit)
A model is useless if people can't interact with it. I wrapped the entire backend logic into a clean, interactive web dashboard.

Live Inference: Users can drag and drop a batch image into the UI, and the Python backend pushes it through the vision_pipeline and PyTorch model in milliseconds.

Dynamic Visual Mapping: I integrated Mermaid.js directly into the Python code. The website actively draws a visual flow-chart of the supply chain that changes color in real-time. If the fruit is fresh, the map turns the delivery routes green. If it's going to rot, it instantly flashes red warnings.
