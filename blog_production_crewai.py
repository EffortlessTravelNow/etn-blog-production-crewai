# Blog Content Production Workflow in CrewAI
# This script implements a complete blog content production workflow using CrewAI
# It orchestrates multiple AI agents to research, write, optimize, and edit blog content

from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain.llms import OpenAI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import json
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app for receiving triggers from n8n
app = FastAPI(title="Blog Content Production API")

# Define request model for blog content creation
class BlogContentRequest(BaseModel):
    blog_topic: str
    primary_keywords: str
    target_audience: str
    call_to_action: str
    desired_word_count: int = 2000
    requester_email: str
    task_id: str = None
    publish_date: str = None

# Initialize OpenAI
llm = OpenAI(temperature=0.7)

# Define tools for agents
google_search_tool = Tool(
    name="Google Search",
    func=lambda query: f"Search results for: {query}",
    description="Useful for searching information on the internet."
)

keyword_analysis_tool = Tool(
    name="Keyword Analysis",
    func=lambda keywords: f"Analysis for keywords: {keywords}",
    description="Analyzes keywords for SEO optimization."
)

# Define the agents
researcher_agent = Agent(
    role="Researcher",
    goal="Conduct thorough research on the blog topic and provide comprehensive insights",
    backstory="""You are an expert researcher with a talent for finding relevant information, 
    statistics, and insights on any topic. You excel at organizing research in a way that 
    writers can easily use.""",
    verbose=True,
    llm=llm,
    tools=[google_search_tool]
)

copywriter_agent = Agent(
    role="Copywriter",
    goal="Write engaging, informative blog content based on research",
    backstory="""You are a skilled copywriter who specializes in creating compelling blog 
    content. You know how to structure articles for readability and engagement while 
    incorporating key messages and research.""",
    verbose=True,
    llm=llm
)

seo_specialist_agent = Agent(
    role="SEO Specialist",
    goal="Optimize content for search engines while maintaining readability",
    backstory="""You are an SEO expert who knows how to optimize content to rank well 
    in search engines. You understand keyword placement, meta descriptions, and content 
    structure for SEO.""",
    verbose=True,
    llm=llm,
    tools=[keyword_analysis_tool]
)

editor_agent = Agent(
    role="Editor",
    goal="Ensure content is polished, error-free, and meets quality standards",
    backstory="""You are a meticulous editor with an eye for detail. You ensure content 
    is grammatically correct, flows well, and meets the client's requirements while 
    maintaining the brand voice.""",
    verbose=True,
    llm=llm
)

# Function to create tasks based on the blog request
def create_tasks(blog_request):
    research_task = Task(
        description=f"""Research the topic: '{blog_request.blog_topic}' thoroughly.
        Focus on the target audience: {blog_request.target_audience}.
        Consider these keywords: {blog_request.primary_keywords}.
        Provide comprehensive research including:
        1. Key facts and statistics
        2. Current trends
        3. Expert opinions
        4. Relevant examples
        5. Keyword suggestions for SEO
        Format your research in a structured way that will be useful for the copywriter.""",
        agent=researcher_agent,
        expected_output="Comprehensive research document with facts, statistics, and insights"
    )
    
    writing_task = Task(
        description=f"""Write a compelling blog post on: '{blog_request.blog_topic}'.
        Target audience: {blog_request.target_audience}
        Primary keywords: {blog_request.primary_keywords}
        Word count: {blog_request.desired_word_count} words
        Call to action: {blog_request.call_to_action}
        
        Use the research provided to create an engaging, informative blog post.
        Structure should include:
        - Attention-grabbing introduction
        - Well-organized body with subheadings
        - Compelling conclusion with the specified call to action
        
        Ensure the content is original, engaging, and valuable to the reader.""",
        agent=copywriter_agent,
        expected_output="Complete blog post draft",
        context=[research_task]
    )
    
    seo_task = Task(
        description=f"""Optimize the blog draft for search engines.
        Primary keywords: {blog_request.primary_keywords}
        
        Review the blog draft and optimize it for SEO by:
        1. Ensuring proper keyword placement and density
        2. Creating an SEO-friendly meta title (under 60 characters)
        3. Writing a compelling meta description (under 160 characters)
        4. Suggesting internal linking opportunities
        5. Optimizing headings and subheadings (H1, H2, H3)
        
        Maintain readability and engagement while optimizing for search engines.""",
        agent=seo_specialist_agent,
        expected_output="SEO-optimized blog post with meta title and description",
        context=[writing_task]
    )
    
    editing_task = Task(
        description=f"""Edit and finalize the blog post.
        Target audience: {blog_request.target_audience}
        Word count: {blog_request.desired_word_count} words
        
        Review the SEO-optimized draft and:
        1. Correct any grammar, spelling, or punctuation errors
        2. Ensure the content flows logically and smoothly
        3. Verify that the blog meets the word count requirement
        4. Check that the call to action is clear and compelling
        5. Ensure the content is properly formatted for web publishing
        
        The final output should be publication-ready and meet professional standards.""",
        agent=editor_agent,
        expected_output="Final, publication-ready blog post",
        context=[seo_task]
    )
    
    return [research_task, writing_task, seo_task, editing_task]

# Function to run the CrewAI workflow
def run_blog_production_workflow(blog_request):
    try:
        logger.info(f"Starting blog production for topic: {blog_request.blog_topic}")
        
        # Create tasks
        tasks = create_tasks(blog_request)
        
        # Create the crew
        crew = Crew(
            agents=[researcher_agent, copywriter_agent, seo_specialist_agent, editor_agent],
            tasks=tasks,
            verbose=2,
            process=Process.sequential
        )
        
        # Run the crew
        result = crew.kickoff()
        
        # Process and structure the result
        structured_result = {
            "task_id": blog_request.task_id or f"BLOG-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "blog_topic": blog_request.blog_topic,
            "target_audience": blog_request.target_audience,
            "primary_keywords": blog_request.primary_keywords,
            "completion_time": datetime.now().isoformat(),
            "research_summary": extract_research_summary(result),
            "blog_draft": extract_blog_content(result),
            "meta_title": extract_meta_title(result),
            "meta_description": extract_meta_description(result)
        }
        
        # Send results back to n8n or to email
        send_results(structured_result, blog_request.requester_email)
        
        # Update Google Sheet if configured
        update_google_sheet(structured_result)
        
        logger.info(f"Blog production completed for topic: {blog_request.blog_topic}")
        return structured_result
        
    except Exception as e:
        logger.error(f"Error in blog production workflow: {str(e)}")
        raise e

# Helper functions for processing results
def extract_research_summary(result):
    # Extract research summary from the full result
    # This would be implemented based on the actual output format
    return "Research summary extracted from results"

def extract_blog_content(result):
    # Extract the final blog content from the full result
    # This would be implemented based on the actual output format
    return "Final blog content extracted from results"

def extract_meta_title(result):
    # Extract the SEO meta title from the full result
    # This would be implemented based on the actual output format
    return "Meta title extracted from results"

def extract_meta_description(result):
    # Extract the SEO meta description from the full result
    # This would be implemented based on the actual output format
    return "Meta description extracted from results"

def send_results(result, email):
    # Send results via email or webhook
    logger.info(f"Sending results to {email}")
    # Implementation would depend on your email service or webhook endpoint

def update_google_sheet(result):
    # Update Google Sheet with the results
    logger.info("Updating Google Sheet with results")
    # Implementation would depend on your Google Sheets API setup

# API endpoints
@app.post("/trigger-blog-production")
async def trigger_blog_production(blog_request: BlogContentRequest, background_tasks: BackgroundTasks):
    """
    Trigger the blog content production workflow
    This endpoint receives requests from n8n and starts the CrewAI process
    """
    logger.info(f"Received request to create blog on topic: {blog_request.blog_topic}")
    
    # Validate request
    if not blog_request.blog_topic:
        raise HTTPException(status_code=400, detail="Blog topic is required")
    
    # Run the workflow in the background
    background_tasks.add_task(run_blog_production_workflow, blog_request)
    
    return {
        "status": "processing",
        "message": f"Blog production started for topic: {blog_request.blog_topic}",
        "task_id": blog_request.task_id or f"BLOG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    }

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Check the status of a blog production task
    """
    # This would be implemented with a database to track task status
    return {"status": "processing", "task_id": task_id}

# Main function to run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
