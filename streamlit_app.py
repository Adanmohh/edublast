import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

def main():
    st.title("Curriculum Processing & Course Generation Demo (Enhanced)")

    # -- 1) Upload PDF Section --
    st.header("1. Upload Curriculum PDF")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file:
        st.write(f"Selected file: {uploaded_file.name}")
        if st.button("Ingest PDF into Qdrant"):
            with st.spinner("Uploading and processing PDF..."):
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                }
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success("PDF uploaded and ingested successfully.")
                        st.experimental_rerun()
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")

    # -- 2) Query the Curriculum --
    st.header("2. Query the Curriculum")
    user_query = st.text_input("Ask a question about the curriculum:")
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
            st.warning("Please enter a question")

    # -- 3) Collections & Generate Entire Course (Old Endpoint) --
    st.header("3. (Optional) Generate Entire Course from a Qdrant Collection")
    collections_response = requests.get(f"{API_URL}/list_curricula")
    selected_collection = None
    if collections_response.status_code == 200:
        data = collections_response.json()
        collections = data.get("collections", [])
        if collections:
            selected_collection = st.selectbox("Select a collection", options=collections)
            if selected_collection:
                debug_resp = requests.get(f"{API_URL}/debug_collection",
                                          params={"collection_name": selected_collection})
                if debug_resp.status_code == 200:
                    coll_data = debug_resp.json()
                    st.write("Documents in this collection:")
                    docs = coll_data.get("documents", [])
                    for d in docs:
                        st.write(f"- {d.get('file_name')}")
        else:
            st.warning("No collections found. Try uploading a PDF.")
    else:
        st.error("Failed to retrieve collections.")

    if selected_collection:
        if st.button("Generate Full Course from Collection"):
            with st.spinner("Generating course..."):
                res = requests.get(
                    f"{API_URL}/generate_course",
                    params={"collection_name": selected_collection}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.subheader("Generated Curriculum")
                    st.json(data)
                else:
                    st.error(f"Generation failed: {res.text}")

    # -- 4) Step-by-Step Course Creation --
    st.header("4. Step-by-Step Course Creation")

    st.markdown("""
    **Flow**:
    1. Create an Outline (title, short desc, duration).
    2. Approve/edit the Outline JSON.
    3. For each module (1-based), propose a lesson idea (optional).
    4. Generate the lesson with that idea, appended to the outline.
    """)

    # 4.1 Create Outline
    st.subheader("Create a New Outline")
    
    # Get available curricula
    collections_response = requests.get(f"{API_URL}/list_curricula")
    available_curricula = []
    if collections_response.status_code == 200:
        data = collections_response.json()
        available_curricula = data.get("collections", [])
    
    with st.form("create_outline_form"):
        # Curriculum selection
        selected_curriculum = st.selectbox(
            "Select Curriculum Base",
            options=["General Knowledge"] + available_curricula,
            help="Choose which curriculum to use as a base for this outline"
        )
        
        course_title = st.text_input("Course Title")
        course_desc = st.text_area("Short Description")
        duration_weeks = st.number_input("Duration in Weeks", min_value=1, step=1, value=4)
        submit_outline_btn = st.form_submit_button("Create Outline")

    if "outline_id" not in st.session_state:
        st.session_state["outline_id"] = None
    if "current_outline" not in st.session_state:
        st.session_state["current_outline"] = {}

    if submit_outline_btn:
        if course_title.strip() and course_desc.strip():
            payload = {
                "title": course_title,
                "short_desc": course_desc,
                "duration_weeks": duration_weeks
            }
            with st.spinner("Creating outline..."):
                try:
                    resp = requests.post(f"{API_URL}/create_course_outline", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["outline_id"] = data["data"]["outline_id"]
                        st.session_state["current_outline"] = data["data"]["outline"]
                        st.success("Outline created successfully!")
                        st.write(f"**Outline ID**: {data['data']['outline_id']}")
                        st.write(f"**Source**: {data['data']['source']}")
                        st.json(data["data"]["outline"])
                    else:
                        st.error(f"Could not create outline: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter both a Title and a Short Description.")

    # 4.2 Approve/Update Outline
    st.subheader("Approve/Update Outline")
    outline_id = st.session_state.get("outline_id")
    current_outline_data = st.session_state.get("current_outline", {})

    if outline_id and current_outline_data:
        st.write(f"**Current Outline ID**: {outline_id}")
        with st.expander("Current Outline JSON", expanded=False):
            st.json(current_outline_data)

        edited_outline_json = st.text_area(
            "Edit Outline JSON before approving (optional)",
            json.dumps(current_outline_data, indent=2),
            height=250
        )
        if st.button("Approve/Update Outline"):
            try:
                updated_outline_dict = json.loads(edited_outline_json)
            except json.JSONDecodeError:
                st.error("Invalid JSON in Outline. Fix and retry.")
                return
            body = {
                "outline_id": outline_id,
                "updated_outline": updated_outline_dict
            }
            resp = requests.post(f"{API_URL}/approve_outline", json=body)
            if resp.status_code == 200:
                data = resp.json()
                st.success("Outline updated successfully.")
                st.session_state["current_outline"] = data["final_outline"]
                st.json(data["final_outline"])
            else:
                st.error(f"Approve outline failed: {resp.text}")
    else:
        st.info("No outline created yet or session state missing. Create one first.")

    # 4.3 Lesson Generation
    st.subheader("Generate Lessons for Outline")
    if outline_id and st.session_state["current_outline"]:
        # Figure out how many modules we have, if any
        modules = st.session_state["current_outline"].get("modules", [])
        if not modules:
            st.warning("No modules found in this outline. Make sure the outline was created properly.")
        else:
            module_options = [f"Module {i+1}: {m['module_title']}" for i,m in enumerate(modules)]
            module_choice = st.selectbox("Select Module to Expand", options=module_options)
            module_number = module_options.index(module_choice) + 1  # 1-based

            # Proposed Lesson Idea
            if "proposed_lesson_idea" not in st.session_state:
                st.session_state["proposed_lesson_idea"] = ""

            if st.button("Propose Next Lesson Idea"):
                # We call our (optional) GET /propose_lesson_prompt
                # If you don't have such an endpoint, skip this step
                with st.spinner("Getting a proposed lesson idea..."):
                    params = {
                        "outline_id": outline_id,
                        "module_number": module_number
                    }
                    try:
                        resp = requests.get(f"{API_URL}/propose_lesson_prompt", params=params)
                        if resp.status_code == 200:
                            data = resp.json()
                            if "proposed_idea" in data:
                                st.session_state["proposed_lesson_idea"] = data["proposed_idea"]
                                st.success("Proposed lesson idea retrieved.")
                            else:
                                st.error("No proposed_idea in response.")
                        else:
                            st.error(f"Proposal request failed: {resp.text}")
                    except Exception as e:
                        st.error(str(e))

            # Text area for lesson prompt
            lesson_idea = st.text_area(
                "Lesson Prompt (modify or type your own)",
                st.session_state.get("proposed_lesson_idea", ""),
                height=100
            )

            if st.button("Generate Lesson for This Module"):
                if lesson_idea.strip():
                    body = {
                        "outline_id": outline_id,
                        "module_index": module_number - 1,  # convert to 0-based for backend
                        "lesson_prompt": lesson_idea,
                    }
                    with st.spinner("Generating lesson..."):
                        resp = requests.post(f"{API_URL}/generate_lesson", json=body)
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("status") == "success":
                                st.success("Lesson generated and appended to outline.")
                                st.write("**New Lesson**:")
                                st.json(data["lesson"])

                                st.write("**Updated Outline**:")
                                st.json(data["updated_outline"])
                                # Update local session
                                st.session_state["current_outline"] = data["updated_outline"]
                                # Clear the proposed lesson idea
                                st.session_state["proposed_lesson_idea"] = ""
                            else:
                                st.error("Lesson generation error.")
                                st.json(data)
                        else:
                            st.error(f"Lesson generation failed: {resp.text}")
                else:
                    st.warning("Enter or propose a lesson idea first.")
    else:
        st.info("No outline loaded or created. Complete the steps above first.")

if __name__ == "__main__":
    main()
