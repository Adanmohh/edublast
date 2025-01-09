import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

def main():
    st.title("Curriculum Processing & Course Generation Demo")

    # PDF Upload
    st.header("1. Upload Curriculum PDF")
    uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])
    if uploaded_file is not None:
        # Show file details
        st.write(f"Selected file: {uploaded_file.name}")
        if st.button("Ingest PDF into Qdrant"):
            with st.spinner("Uploading and processing PDF..."):
                # Convert to bytes and send
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success("PDF uploaded and ingested successfully.")
                        # Force refresh of curricula list
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")

    st.header("2. Query the Curriculum")
    user_query = st.text_input("Enter your question about the curriculum:")
    if st.button("Query Curriculum"):
        if user_query.strip():
            res = requests.get(f"{API_URL}/query", params={"question": user_query})
            if res.status_code == 200:
                data = res.json()
                if "answer" in data:
                    st.write(f"**Answer**: {data['answer']}")
                else:
                    st.write(data)
            else:
                st.error(f"Query failed: {res.text}")
        else:
            st.warning("Please enter a question.")

    st.header("3. Select Curriculum & Generate Course")
    
    # Get available collections
    collections_response = requests.get(f"{API_URL}/list_curricula")
    if collections_response.status_code == 200:
        data = collections_response.json()
        collections = data.get("collections", [])
        if collections:
            selected_collection = st.selectbox(
                "Select a collection",
                options=collections
            )
            if selected_collection:
                try:
                    # Get collection contents
                    debug_response = requests.get(
                        f"{API_URL}/debug_collection",
                        params={"collection_name": selected_collection}
                    )
                    if debug_response.status_code == 200:
                        collection_data = debug_response.json()
                        
                        # Display collection overview
                        st.subheader("Collection Contents")
                        st.write("This collection contains the following documents:")
                        
                        # Extract and display document info
                        if "documents" in collection_data:
                            documents = collection_data["documents"]
                            if documents:
                                st.write(f"Found {len(documents)} documents:")
                                for doc in documents:
                                    st.write(f"- {doc['file_name']}")
                                
                                st.info("""
                                The documents in this collection are ready for:
                                1. Querying - Ask questions about the curriculum content
                                2. Course Generation - Create a structured course outline
                                
                                Select an operation above or press 'Generate Course' to create a course structure.
                                """)
                            else:
                                st.error("No documents found in collection")
                        else:
                            st.error("Invalid collection data format")
                            
                except Exception as e:
                    st.error(f"Error loading collection details: {str(e)}")
        else:
            st.warning("No collections available. Please upload a PDF first.")
            selected_collection = None
    else:
        st.error("Failed to fetch collections")
        selected_collection = None

    def display_curriculum(data):
        """Display the curriculum structure in a hierarchical format."""
        if "error" in data:
            st.error(data["error"])
            return

        # Main curriculum info
        st.header(data["title"])
        st.write(data["overview"])
        
        # Target audience and goals
        st.subheader("Course Information")
        st.write(f"**Target Audience:** {data['target_audience']}")
        
        if data["learning_goals"]:
            st.write("**Learning Goals:**")
            for goal in data["learning_goals"]:
                st.write(f"- {goal}")
        
        # Modules
        for module in data["modules"]:
            st.subheader(f"📚 {module['title']}")
            st.write(module["description"])
            
            # Lessons in this module
            for lesson in module["lessons"]:
                st.markdown(f"### 📖 {lesson['title']}")
                
                # Lesson details in expandable sections
                with st.expander("View Lesson Details"):
                    if lesson["objectives"]:
                        st.write("**Learning Objectives:**")
                        for obj in lesson["objectives"]:
                            st.write(f"- {obj}")
                            
                    if lesson["key_points"]:
                        st.write("**Key Points:**")
                        for point in lesson["key_points"]:
                            st.write(f"- {point}")
                            
                    if lesson["activities"]:
                        st.write("**Activities:**")
                        for activity in lesson["activities"]:
                            st.write(f"- {activity}")
                            
                    if lesson["assessment_ideas"]:
                        st.write("**Assessment Ideas:**")
                        for idea in lesson["assessment_ideas"]:
                            st.write(f"- {idea}")
                            
                    if lesson["resources"]:
                        st.write("**Resources:**")
                        for resource in lesson["resources"]:
                            st.write(f"- {resource}")

    if st.button("Generate Course") and selected_collection:
        with st.spinner("Generating course structure from collection..."):
            # First get the collection contents to verify we have documents
            debug_response = requests.get(
                f"{API_URL}/debug_collection",
                params={"collection_name": selected_collection}
            )
            if debug_response.status_code != 200:
                st.error("Failed to verify collection contents")
                return
                
            # Generate course structure using all documents in collection
            res = requests.get(
                f"{API_URL}/generate_course",
                params={"collection_name": selected_collection}
            )
            if res.status_code == 200:
                data = res.json()
                st.subheader("Generated Curriculum Structure")
                st.markdown("---")
                display_curriculum(data)
                st.markdown("---")
            else:
                st.error(f"Course generation failed: {res.text}")

if __name__ == "__main__":
    main()
