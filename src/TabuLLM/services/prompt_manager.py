import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

class PromptManager:
    """Manages LLM prompts using Jinja2 templates."""
    
    def __init__(self, template_dir=None):
        if template_dir is None:
            # Default to the 'prompts' directory relative to this file's parent
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
        
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
        
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

# Singleton instance
prompt_manager = PromptManager()
