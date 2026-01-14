import logging
import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from llm_core.factory import LLMFactory
import re

logger = logging.getLogger("summary_generator")

class SummaryGenerator:
    def __init__(self):
        self.llm_provider = LLMFactory.get_provider()
        self.llm = self.llm_provider.get_base_model()

    

    
    def generate_general_section(self, meeting_data: Dict, participants: List[Dict]) -> str:
        """
        Generates a formal markdown section for meeting details.
        """

        def format_participants(participants: list[dict]) -> str:
            """
            Formats participants into deterministic Markdown
            compatible with ngx-markdown and the locked General Section prompt.
            """

            if not participants:
                return "> No participants recorded."

            lines = []
            for p in participants:
                name = p.get("first_name",'') +' '+ p.get("last_name",'')  or "N/A"
                email = p.get("email")
        

                # Build display name
                display_name = name
                if email:
                    display_name = f"{name}, {email}"

                lines.append(
                    f"> - **{display_name}** "
                )

            return "\n".join(lines)
        
        prompt = ChatPromptTemplate.from_template("""
            You are a top-tier Corporate Secretary preparing formal meeting records.

            Task:
            Generate the General / Meeting Details section in pure Markdown.

            IMPORTANT:
            - The heading marker for ALL section titles is EXACTLY: ####
            - NEVER use ##### or ###
            - DO NOT adjust heading levels
            - DO NOT repeat or infer heading markers

            MANDATORY OUTPUT TEMPLATE (COPY EXACTLY):

            #### **Meeting Title**

            > {name}

            #### **Meeting Information**

            > - **Meeting ID**: {id}
            > - **Date**: {date}
            > - **Group**: {group}
            > - **Department**: {dept}
            > - **Venue**: {venue}

            #### **Attendees**

            {participants_md}

            Rules:
            - Preserve the heading markers exactly as shown above
            - Always include a blank line before and after headings
            - Use > blockquotes for all body content
            - Do NOT use HTML
            - Do NOT wrap output in code blocks
            - Return ONLY raw Markdown
            """)
        
        try:
            chain = prompt | self.llm

            response = chain.invoke({
                "id": meeting_data.get("meeting_id", "N/A"),
                "name": meeting_data.get("meeting_name", "N/A"),
                "date": meeting_data.get("scheduled_at", "N/A"),
                "group": meeting_data.get("group_name", "N/A"),
                "dept": meeting_data.get("department", "N/A"),
                "venue": meeting_data.get("venue", "N/A"),
                "participants_md": format_participants(participants)
            })
            markdown = re.sub(r"^#{5,}\s+", "#### ", response.content.strip(), flags=re.MULTILINE)
            return markdown
        except Exception as e:
            logger.error(f"Error generating general section: {e}")
            return "# Critical Error\nCould not generate meeting details."



    def generate_deep_analysis(self, agenda_item: str, context_map: Dict[str, List[str]]) -> str:
        """
        Generates a deep dive analysis for a single agenda item using grouped context.
        """
        
        # Prepare context string
        context_str = ""
        if not context_map:
             context_str = "No specific documents found."
        else:
            for doc_type, chunks in context_map.items():
                context_str += f"\n\n### Source: {doc_type}\n"
                for i, chunk in enumerate(chunks[:5]): # limit chunks to avoid context overflow
                    context_str += f"- {chunk}\n"

        PROMPT_TEMPLATE = """
        You are an expert Corporate Analyst preparing a formal Board Meeting Record.

        Agenda Item: "{agenda_item}"

        Available Context / Evidence:
        {context_str}

        Task:
        Write a deep, formal analysis of the agenda item in pure Markdown.

        Mandatory Structure (STRICT):

        #### **Summary of Discussion**
        > Describe what was discussed. Derive strictly from MOM or minutes.

        #### **Key Evidence**
        > List concrete data points, figures, or observations.
        > Cite source types in brackets (e.g., [MOM], [Policy], [Audit Report]).

        #### **Decisions Taken**
        > Document approvals, resolutions, or rejections.

        #### **Action Steps**
        > List ATRs or newly agreed actions.

        Formatting Rules (CRITICAL):
        - Use #### for section headings
        - Use > blockquotes for ALL body content
        - Use Markdown lists inside blockquotes where needed
        - Do NOT use HTML
        - Do NOT wrap output in code blocks
        - Do NOT infer or fabricate information
        - If information is missing, write: No specific record found

        Tone:
        - Professional
        - Objective
        - Board-level

        Return ONLY raw Markdown text.
        """

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "agenda_item": agenda_item,
                "context_str": context_str
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating deep analysis for '{agenda_item}': {e}")
            return f"Error generating analysis for item: {agenda_item}"

