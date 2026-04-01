import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils.logging_config import logger

class PromptManager:
    """Manages LLM prompts using Jinja2 templates."""
    
    def __init__(self, template_dir=None):
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
        
        logger.info(f"PromptManager: Initializing with template directory: {template_dir}")
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def get_prompt(self, template_name, **kwargs):
        """Loads a template and renders it with the provided keyword arguments."""
        if not template_name.endswith('.j2'):
            template_name += '.j2'
        
        logger.debug(f"PromptManager: Loading template '{template_name}'.")
        template = self.env.get_template(template_name)
        rendered = template.render(**kwargs)
        logger.info(f"PromptManager: Successfully rendered '{template_name}'.")
        return rendered

# Singleton instance
prompt_manager = PromptManager()
