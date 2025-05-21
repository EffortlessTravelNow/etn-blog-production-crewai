"""
Simplified Blog Production API
This script provides a simplified version of the blog production workflow
using direct OpenAI API calls instead of the full CrewAI framework
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import openai
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Simplified Blog Production API")

# Get OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OPENAI_API_KEY environment variable not set")

# Define request model
class BlogContentRequest(BaseModel):
    blog_topic: str
    primary_keywords: str
    target_audience: str
    call_to_action: str
    desired_word_count: int = 4000
    requester_email: str
    task_id: str = None
    publish_date: str = None

# Function to generate research content
async def generate_research(blog_request):
    try:
        prompt = f"""
        Research the topic: '{blog_request.blog_topic}' thoroughly.
        Focus on the target audience: {blog_request.target_audience}.
        Consider these keywords: {blog_request.primary_keywords}.
        
        Provide comprehensive research including:
        1. Key facts and statistics
        2. Current trends
        3. Expert opinions
        4. Relevant examples
        5. Keyword suggestions for SEO
        
        Format your research in a structured way that will be useful for writing a blog post.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert researcher with a talent for finding relevant information, statistics, and insights on any topic."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating research: {str(e)}")
        return f"Error generating research: {str(e)}"

# Function to write blog content
async def write_blog_content(blog_request, research):
    try:
        prompt = f"""
        Write a compelling blog post on: '{blog_request.blog_topic}'.
        Target audience: {blog_request.target_audience}
        Primary keywords: {blog_request.primary_keywords}
        Word count: {blog_request.desired_word_count} words
        Call to action: {blog_request.call_to_action}
        
        Use this research to create an engaging, informative blog post:
        {research}
        
        Structure should include:
        - Attention-grabbing introduction
        - Well-organized body with subheadings
        - Compelling conclusion with the specified call to action
        
        Ensure the content is original, engaging, and valuable to the reader.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled copywriter who specializes in creating compelling blog content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error writing blog content: {str(e)}")
        return f"Error writing blog content: {str(e)}"

# Function to optimize content for SEO
async def optimize_for_seo(blog_request, blog_content):
    try:
        prompt = f"""
        Optimize this blog draft for search engines:
        
        {blog_content}
        
        Primary keywords: {blog_request.primary_keywords}
        
        Please provide:
        1. An SEO-friendly meta title (under 60 characters)
        2. A compelling meta description (under 160 characters)
        3. The optimized blog content with proper keyword placement
        
        Format your response as:
        META TITLE: [Your meta title here]
        META DESCRIPTION: [Your meta description here]
        
        OPTIMIZED CONTENT:
        [The optimized blog content]
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an SEO expert who knows how to optimize content to rank well in search engines."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error optimizing for SEO: {str(e)}")
        return f"Error optimizing for SEO: {str(e)}"

# Function to edit and finalize content
async def edit_and_finalize(blog_request, seo_content):
    try:
        prompt = f"""
        Edit and finalize this blog post:
        
        {seo_content}
        
        Target audience: {blog_request.target_audience}
        Word count: {blog_request.desired_word_count} words
        
        Please:
        1. Correct any grammar, spelling, or punctuation errors
        2. Ensure the content flows logically and smoothly
        3. Verify that the blog meets the word count requirement
        4. Check that the call to action is clear and compelling
        5. Ensure the content is properly formatted for web publishing
        
        The final output should be publication-ready and meet professional standards.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a meticulous editor with an eye for detail."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error editing and finalizing: {str(e)}")
        return f"Error editing and finalizing: {str(e)}"

# Function to extract meta title and description
def extract_meta_data(seo_content):
    meta_title = ""
    meta_description = ""
    
    lines = seo_content.split('\n')
    for line in lines:
        if line.startswith("META TITLE:"):
            meta_title = line.replace("META TITLE:", "").strip()
        elif line.startswith("META DESCRIPTION:"):
            meta_description = line.replace("META DESCRIPTION:", "").strip()
    
    return meta_title, meta_description

# Function to run the blog production workflow
async def run_blog_production_workflow(blog_request):
    try:
        logger.info(f"Starting blog production for topic: {blog_request.blog_topic}")
        
        # Step 1: Generate research
        research = await generate_research(blog_request)
        logger.info("Research generation complete")
        
        # Step 2: Write blog content
        blog_content = await write_blog_content(blog_request, research)
        logger.info("Blog content writing complete")
        
        # Step 3: Optimize for SEO
        seo_content = await optimize_for_seo(blog_request, blog_content)
        logger.info("SEO optimization complete")
        
        # Step 4: Edit and finalize
        final_content = await edit_and_finalize(blog_request, seo_content)
        logger.info("Editing and finalizing complete")
        
        # Extract meta title and description
        meta_title, meta_description = extract_meta_data(seo_content)
        
        # Process and structure the result
        structured_result = {
            "task_id": blog_request.task_id or f"BLOG-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "blog_topic": blog_request.blog_topic,
            "target_audience": blog_request.target_audience,
            "primary_keywords": blog_request.primary_keywords,
            "completion_time": datetime.now().isoformat(),
            "research_summary": research,
            "blog_draft": final_content,
            "meta_title": meta_title,
            "meta_description": meta_description
        }
        
        # Log completion
        logger.info(f"Blog production completed for topic: {blog_request.blog_topic}")
        return structured_result
        
    except Exception as e:
        logger.error(f"Error in blog production workflow: {str(e)}")
        raise e

# API endpoints
@app.post("/trigger-blog-production")
async def trigger_blog_production(blog_request: BlogContentRequest, background_tasks: BackgroundTasks):
    """
    Trigger the blog content production workflow
    This endpoint receives requests from n8n and starts the process
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

@app.get("/")
async def root():
    """
    Root endpoint to verify the API is running
    """
    return {"status": "online", "message": "Simplified Blog Production API is running"}

# Main function to run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
