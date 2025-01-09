import streamlit as st
import requests
import json

# Adjust if your FastAPI service is running elsewhere
API_URL = "http://localhost:8000"

def main():
    st.title("Curriculum Processing & Course Generation Demo")

    # -------------------------------------------------------------
    # 1. Upload Curriculum PDF
    # -------------------------------------------------------------
    st.header("1. Upload Curriculum PDF")
    uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])
    if uploaded_file is not None:
        st.write(f"Selected file: {uploaded_file.name}")
        if st.button("Ingest PDF into Qdrant"):
            with st.spinner("Uploading and processing PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success("PDF uploaded and ingested successfully.")
                        st.experimental_rerun()
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")

    # -------------------------------------------------------------
    # 2. Query the Curriculum
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 3. Collections & Generate Whole Course (Old Endpoint)
    # -------------------------------------------------------------
    st.header("3. Select Curriculum & (Optional) Generate Entire Course from All Docs")

    # Fetch available collections
    collections_response = requests.get(f"{API_URL}/list_curricula")
    selected_collection = None
    if collections_response.status_code == 200:
        data = collections_response.json()
        collections = data.get("collections", [])
        if collections:
            selected_collection = st.selectbox("Select a Qdrant collection", options=collections)
            if selected_collection:
                # Show debug info about the selected collection
                debug_response = requests.get(
                    f"{API_URL}/debug_collection",
                    params={"collection_name": selected_collection}
                )
                if debug_response.status_code == 200:
                    collection_data = debug_response.json()
                    st.subheader("Collection Contents")
                    if "documents" in collection_data:
                        documents = collection_data["documents"]
                        if documents:
                            st.write(f"Found {len(documents)} document(s):")
                            for doc in documents:
                                st.write(f"- {doc['file_name']}")
                        else:
                            st.warning("No documents found in this collection.")
                    else:
                        st.error("Invalid data from debug_collection.")
                else:
                    st.error("Could not load debug info for this collection.")
        else:
            st.warning("No collections found. Upload a PDF first.")
    else:
        st.error("Failed to fetch collections.")

    def display_curriculum(curriculum_data):
        """Pretty-print the automatically generated curriculum in a hierarchical format."""
        if "error" in curriculum_data:
            st.error(curriculum_data["error"])
            return

        st.header(curriculum_data.get("title", "Untitled Curriculum"))
        st.write(curriculum_data.get("overview", "No overview provided."))
        st.write(f"**Target Audience:** {curriculum_data.get('target_audience', 'N/A')}")
        goals = curriculum_data.get("learning_goals", [])
        if goals:
            st.write("**Learning Goals:**")
            for g in goals:
                st.write(f"- {g}")

        modules = curriculum_data.get("modules", [])
        for m_idx, module in enumerate(modules):
            st.subheader(f"Module {m_idx+1}: {module.get('title','(No title)')}")
            st.write(module.get("description","(No description)"))
            for l_idx, lesson in enumerate(module.get("lessons", [])):
                st.markdown(f"### Lesson {l_idx+1}: {lesson.get('title','(No title)')}")
                with st.expander("Lesson Details", expanded=False):
                    # Show each section
                    if "objectives" in lesson and lesson["objectives"]:
                        st.write("**Objectives:**")
                        for obj in lesson["objectives"]:
                            st.write(f"- {obj}")

                    if "key_points" in lesson and lesson["key_points"]:
                        st.write("**Key Points:**")
                        for kp in lesson["key_points"]:
                            st.write(f"- {kp}")

                    if "activities" in lesson and lesson["activities"]:
                        st.write("**Activities:**")
                        for act in lesson["activities"]:
                            st.write(f"- {act}")

                    if "assessment_ideas" in lesson and lesson["assessment_ideas"]:
                        st.write("**Assessment Ideas:**")
                        for idea in lesson["assessment_ideas"]:
                            st.write(f"- {idea}")

                    if "resources" in lesson and lesson["resources"]:
                        st.write("**Resources:**")
                        for rsc in lesson["resources"]:
                            st.write(f"- {rsc}")

    if st.button("Generate Full Course from Collection") and selected_collection:
        with st.spinner("Generating course structure from collection..."):
            res = requests.get(
                f"{API_URL}/generate_course",
                params={"collection_name": selected_collection}
            )
            if res.status_code == 200:
                data = res.json()
                st.subheader("Generated Curriculum Structure")
                display_curriculum(data)
            else:
                st.error(f"Generation failed: {res.text}")

    # -------------------------------------------------------------
    # 4. Step-by-Step Course Creation Endpoints
    # -------------------------------------------------------------
    st.header("4. Step-by-Step Course Creation")

    st.markdown("""
        Here you can create a custom course outline (title, description, duration)
        step by step, then generate lessons one at a time. 
    """)

    # 4.1 Create Outline
    st.subheader("Create Outline")
    with st.form(key="create_outline_form"):
        course_title = st.text_input("Course Title", "")
        course_description = st.text_area("Short Description", "")
        duration_weeks = st.number_input("Duration (weeks)", min_value=1, step=1, value=4)
        create_outline_submitted = st.form_submit_button("Create Course Outline")

    if create_outline_submitted:
        if course_title.strip() and course_description.strip():
            with st.spinner("Generating outline..."):
                payload = {
                    "title": course_title,
                    "short_desc": course_description,
                    "duration_weeks": duration_weeks
                }
                try:
                    resp = requests.post(f"{API_URL}/create_course_outline", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        if "outline_id" in data:
                            st.session_state["outline_id"] = data["outline_id"]
                            st.session_state["current_outline"] = data["outline"]
                            st.success(f"Outline created with ID {data['outline_id']}")
                            st.json(data["outline"])
                        else:
                            st.error("No outline_id found in the response.")
                            st.json(data)
                    else:
                        st.error(f"Failed to create outline: {resp.text}")
                except Exception as e:
                    st.error(f"Error creating outline: {str(e)}")
        else:
            st.warning("Please fill in both Title and Description.")

    # 4.2 Approve/Update Outline
    st.subheader("Approve or Edit Outline")
    outline_id = st.session_state.get("outline_id")
    current_outline_data = st.session_state.get("current_outline")

    if outline_id and current_outline_data:
        st.write(f"**Current Outline ID**: {outline_id}")
        with st.expander("Outline JSON", expanded=False):
            st.json(current_outline_data)

        updated_outline_str = st.text_area(
            "Optionally edit the JSON outline before approving:",
            json.dumps(current_outline_data, indent=2)
        )

        if st.button("Approve/Update Outline"):
            try:
                parsed_outline = json.loads(updated_outline_str)
            except json.JSONDecodeError:
                st.error("Invalid JSON. Fix the JSON and try again.")
                return

            # call /approve_outline
            body = {
                "outline_id": outline_id,
                "updated_outline": parsed_outline
            }
            resp = requests.post(f"{API_URL}/approve_outline", json=body)
            if resp.status_code == 200:
                data = resp.json()
                st.success("Outline approved/updated successfully.")
                st.session_state["current_outline"] = data["final_outline"]
                st.json(data["final_outline"])
            else:
                st.error(f"Approve outline failed: {resp.text}")
    else:
        st.info("Create a new outline or select an existing outline to approve/edit.")

    # 4.3 Generate Single Lesson
    st.subheader("Generate One Lesson")
    if outline_id and current_outline_data:
        module_count = len(current_outline_data.get("modules", []))
        if module_count == 0:
            st.warning("No modules found in the current outline.")
        else:
            module_index = st.number_input(
                "Module Index to Expand (0-based)", min_value=0, max_value=module_count - 1
            )
            lesson_prompt = st.text_area("Lesson Prompt (e.g., 'Cover advanced Python OOP')")
            if st.button("Generate Lesson for Module"):
                if lesson_prompt.strip():
                    body = {
                        "outline_id": outline_id,
                        "module_index": module_index,
                        "lesson_prompt": lesson_prompt,
                    }
                    with st.spinner("Generating lesson..."):
                        resp = requests.post(f"{API_URL}/generate_lesson", json=body)
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("status") == "success":
                                st.success("Lesson generated successfully!")
                                st.write("**New Lesson:**")
                                st.json(data["lesson"])
                                st.write("**Updated Outline:**")
                                st.json(data["updated_outline"])
                                # Update local session state
                                st.session_state["current_outline"] = data["updated_outline"]
                            else:
                                st.error("Lesson generation error.")
                                st.json(data)
                        else:
                            st.error(f"Failed to generate lesson: {resp.text}")
                else:
                    st.warning("Please provide a lesson prompt.")
    else:
        st.info("No outline to generate lessons for. Create and approve an outline first.")

if __name__ == "__main__":
    main()
