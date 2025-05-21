"""
Enhanced Blog Production API with ETN Brand Guidelines
This script provides an improved version of the blog production workflow
using direct OpenAI API calls with enhanced prompts for brand-aligned, SEO-optimized content
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
app = FastAPI(title="ETN Enhanced Blog Production API")

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
    desired_word_count: int = 4000  # Increased default word count
    requester_email: str
    task_id: str = None
    publish_date: str = None

# ETN Brand Voice Guidelines
ETN_BRAND_VOICE = """
ETN Brand Voice Guidelines:
- Friendly yet professional: Write as a trusted travel companion, not a salesperson
- Inclusive and accessible: Use language that welcomes all travelers, especially those with accessibility needs
- Practical and actionable: Provide specific, useful advice rather than generic statements
- Warm and inspiring: Encourage travel while acknowledging challenges some travelers face
- Authoritative but not condescending: Demonstrate expertise without talking down to readers
- Avoid clichés and overly enthusiastic language like "amazing," "incredible," or "must-see"
- Use clear, straightforward language that avoids jargon
- Never end with generic phrases like "So buckle up and start planning your dream vacation!"
- Focus on practical information, authentic experiences, and inclusive travel options
"""

# ETN SEO Guidelines
ETN_SEO_GUIDELINES = """
ETN SEO Guidelines:
- Focus on organic search optimization only (no paid advertising strategies)
- Follow white-hat SEO practices exclusively
- Target keywords must be naturally integrated throughout the content
- Include primary keywords in title, meta description, H1, and at least one H2
- Structure content with proper heading hierarchy (H1, H2, H3)
- Create comprehensive content that thoroughly addresses the search intent
- Include internal linking opportunities to relevant ETN content
- Ensure all images have descriptive alt text for accessibility and SEO
- Focus on underserved traveler groups (accessible travel, senior travelers, pet owners, etc.)
- Provide specific, actionable advice rather than general statements
- Address common questions and pain points from the target audience
"""

# Function to generate research content
async def generate_research(blog_request):
    try:
        prompt = f"""
        Research the topic: '{blog_request.blog_topic}' thoroughly and comprehensively.
        Focus on the target audience: {blog_request.target_audience}.
        Consider these keywords: {blog_request.primary_keywords}.
        
        Provide extensive, in-depth research including:
        1. Key facts and statistics with credible sources
        2. Current trends and future predictions in the travel industry
        3. Expert opinions and quotes from travel authorities
        4. Detailed case studies and relevant examples of real travel experiences
        5. Comprehensive keyword analysis for SEO (primary and related long-tail keywords)
        6. Common questions and pain points from the target audience
        7. Specific, actionable advice for travelers (not generic statements)
        
        For ETN's audience specifically:
        - If researching accessible travel: Include specific accessibility features, regulations, and accommodations
        - If researching pet travel: Include pet policies, requirements, and pet-friendly accommodations
        - If researching family travel: Include child-friendly activities, safety considerations, and family discounts
        - If researching senior travel: Include mobility considerations, senior discounts, and medical information
        
        Format your research in a highly structured way with clear sections, subsections, and bullet points.
        Include at least 10-15 key insights that will make this blog post stand out as authoritative and valuable.
        Aim for depth and comprehensiveness - this research will be used to create a detailed {blog_request.desired_word_count}-word blog post.
        
        Remember that ETN (Effortless Travel Now) focuses on making travel accessible and enjoyable for all travelers, 
        with special emphasis on those with accessibility needs, pet owners, families, and seniors.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are an expert travel researcher with a talent for finding relevant information, statistics, and insights on travel topics. You provide comprehensive, well-structured research that goes beyond surface-level information. You focus on practical, actionable information that helps travelers, especially those with specific needs like accessibility requirements."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
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
        Write a comprehensive, in-depth blog post on: '{blog_request.blog_topic}'.
        Target audience: {blog_request.target_audience}
        Primary keywords: {blog_request.primary_keywords}
        Word count: {blog_request.desired_word_count} words (aim for at least {blog_request.desired_word_count} words)
        Call to action: {blog_request.call_to_action}
        
        Use this research to create an authoritative, valuable blog post:
        {research}
        
        Structure requirements:
        - Compelling, attention-grabbing introduction that establishes the importance of the topic
        - At least 7-10 main sections with descriptive headings (H2s) that include target keywords
        - Multiple subsections (H3s) under each main section
        - Detailed examples, case studies, and actionable advice in each section
        - Expert quotes or statistics with sources to support key points
        - Visual element suggestions (images, infographics, charts) at appropriate points
        - Comprehensive conclusion that summarizes key takeaways
        - Strong call to action: {blog_request.call_to_action}
        
        Content quality requirements:
        - Provide specific, actionable advice rather than general statements
        - Include step-by-step instructions where appropriate
        - Address common questions and objections from the target audience
        - Use a conversational yet authoritative tone
        - Incorporate storytelling elements to engage readers
        - Ensure content is original, engaging, and provides exceptional value
        - Focus on practical information that helps readers solve real travel problems
        
        {ETN_BRAND_VOICE}
        
        This should be a definitive guide that thoroughly covers all aspects of the topic and positions ETN as an authority in travel, especially for travelers with specific needs.
        
        IMPORTANT: Avoid ending with generic phrases like "So buckle up and start planning your dream vacation!" Instead, end with specific, actionable advice related to the topic and a natural transition to the call to action.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a skilled travel copywriter for Effortless Travel Now (ETN), specializing in creating comprehensive, engaging blog content that provides exceptional value to travelers. You excel at creating long-form, authoritative content that ranks well in search engines while maintaining ETN's friendly, inclusive, and practical brand voice. You focus on making travel accessible and enjoyable for all travelers, with special emphasis on those with accessibility needs, pet owners, families, and seniors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
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
        Optimize this blog draft for search engines following ETN's SEO guidelines:
        
        {blog_content}
        
        Primary keywords: {blog_request.primary_keywords}
        Target audience: {blog_request.target_audience}
        
        {ETN_SEO_GUIDELINES}
        
        Please provide:
        1. An SEO-friendly meta title (under 60 characters) that includes the primary keyword and is compelling for clicks
        2. A compelling meta description (under 160 characters) that drives clicks and includes the primary keyword
        3. The optimized blog content with:
           - Proper keyword placement (title, headings, first paragraph, throughout content)
           - Optimized heading structure (H1, H2, H3)
           - Internal linking suggestions (at least 3-5 opportunities to link to other ETN content)
           - External linking suggestions to authoritative sources
           - Image alt text suggestions that include relevant keywords and describe images for accessibility
        
        Format your response as:
        META TITLE: [Your meta title here]
        META DESCRIPTION: [Your meta description here]
        
        OPTIMIZED CONTENT:
        [The optimized blog content]
        
        Ensure the content remains natural and reader-friendly while being optimized for search engines.
        The content should maintain ETN's brand voice while incorporating SEO best practices.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are an SEO expert for Effortless Travel Now (ETN) who specializes in optimizing travel content to rank well in search engines while maintaining readability and user engagement. You follow white-hat SEO practices exclusively and focus on organic search optimization. You understand the importance of making content accessible to all users, including those with disabilities, and recognize that good accessibility practices often align with good SEO practices."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
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
        Edit and finalize this blog post to professional publishing standards for Effortless Travel Now (ETN):
        
        {seo_content}
        
        Target audience: {blog_request.target_audience}
        Word count: {blog_request.desired_word_count} words
        
        {ETN_BRAND_VOICE}
        
        Please:
        1. Correct any grammar, spelling, or punctuation errors
        2. Ensure the content flows logically and smoothly
        3. Verify that the blog meets the word count requirement of at least {blog_request.desired_word_count} words
        4. Check that the call to action is clear, compelling, and strategically placed
        5. Ensure the content is properly formatted for web publishing with:
           - Consistent heading hierarchy
           - Short, scannable paragraphs (3-4 sentences maximum)
           - Bulleted or numbered lists where appropriate
           - Proper use of bold and italics for emphasis
           - Transition phrases between sections
        6. Add engaging subheadings if needed
        7. Suggest pull quotes or highlight sections
        8. Ensure all advice is practical and actionable
        9. Remove any generic travel clichés or overly enthusiastic language
        10. Verify that the conclusion is strong and leads naturally to the call to action
        
        The final output should be publication-ready, comprehensive, and meet professional standards.
        It should exemplify ETN's brand voice: friendly yet professional, inclusive, practical, and authoritative without being condescending.
        
        IMPORTANT: Check the ending carefully to ensure it doesn't use generic phrases like "So buckle up and start planning your dream vacation!" 
        Instead, it should end with specific, actionable advice related to the topic and a natural transition to the call to action.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a meticulous editor for Effortless Travel Now (ETN) with an eye for detail who ensures content is polished, comprehensive, and ready for publication. You maintain ETN's brand voice: friendly yet professional, inclusive, practical, and authoritative without being condescending. You focus on making travel content accessible and valuable to all travelers, with special emphasis on those with accessibility needs, pet owners, families, and seniors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
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
        logger.info(f"Starting enhanced blog production for topic: {blog_request.blog_topic}")
        
        # Step 1: Generate research
        logger.info("Starting research generation")
        research = await generate_research(blog_request)
        logger.info("Research generation complete")
        
        # Step 2: Write blog content
        logger.info("Starting blog content writing")
        blog_content = await write_blog_content(blog_request, research)
        logger.info("Blog content writing complete")
        
        # Step 3: Optimize for SEO
        logger.info("Starting SEO optimization")
        seo_content = await optimize_for_seo(blog_request, blog_content)
        logger.info("SEO optimization complete")
        
        # Step 4: Edit and finalize
        logger.info("Starting editing and finalizing")
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
            "meta_description": meta_description,
            "word_count": len(final_content.split())
        }
        
        # Log completion
        logger.info(f"Enhanced blog production completed for topic: {blog_request.blog_topic}")
        return structured_result
        
    except Exception as e:
        logger.error(f"Error in blog production workflow: {str(e)}")
        raise e

# API endpoints
@app.post("/trigger-blog-production")
async def trigger_blog_production(blog_request: BlogContentRequest, background_tasks: BackgroundTasks):
    """
    Trigger the enhanced blog content production workflow
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
        "message": f"Enhanced blog production started for topic: {blog_request.blog_topic}",
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
    return {"status": "online", "message": "ETN Enhanced Blog Production API is running"}

# Main function to run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
