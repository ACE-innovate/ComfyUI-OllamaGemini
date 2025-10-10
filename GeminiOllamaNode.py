import os

import google.generativeai as genai
from PIL import Image
import torch
import numpy as np

# Common function to apply prompt structure templates
def apply_prompt_template(prompt, prompt_structure="Custom"):
    # Define prompt structure templates
    prompt_templates = {
        "VideoGen": "Create a professional cinematic video generation prompt based on my description. Structure your prompt in this precise order: (1) SUBJECT: Define main character(s)/object(s) with specific, vivid details (appearance, expressions, attributes); (2) CONTEXT/SCENE: Establish the detailed environment with atmosphere, time of day, weather, and spatial relationships; (3) ACTION: Describe precise movements and temporal flow using dynamic verbs and sequential language ('first... then...'); (4) CINEMATOGRAPHY: Specify exact camera movements (dolly, pan, tracking), shot types (close-up, medium, wide), lens choice (35mm, telephoto), and professional lighting terminology (Rembrandt, golden hour, backlit); (5) STYLE: Define the visual aesthetic using specific references to film genres, directors, or animation styles. For realistic scenes, emphasize photorealism with natural lighting and physics. For abstract/VFX, include stylistic terms (surreal, psychedelic) and dynamic descriptors (swirling, morphing). For animation, specify the exact style (anime, 3D cartoon, hand-drawn). Craft a single cohesive paragraph that flows naturally while maintaining technical precision. Return ONLY the prompt text itself no more 200 tokens.",

        "FLUX.1-dev": "As an elite text-to-image prompt engineer, craft an exceptional FLUX.1-dev prompt from my description. Create a hyper-detailed, cinematographic paragraph that includes: (1) precise subject characterization with emotional undertones, (2) specific artistic influences from legendary painters/photographers, (3) technical camera specifications (lens, aperture, perspective), (4) sophisticated lighting setup with exact quality and direction, (5) atmospheric elements and depth effects, (6) composition techniques, and (7) post-processing styles. Use language that balances technical precision with artistic vision. Return ONLY the prompt text itself - no explanations or formatting no more 200 tokens.",

        "SDXL": "Create a premium comma-separated tag prompt for SDXL based on my description. Structure the prompt with these elements in order of importance: (1) main subject with precise descriptors, (2) high-impact artistic medium (oil painting, digital art, photography, etc.), (3) specific art movement or style with named influences, (4) professional lighting terminology (rembrandt, cinematic, golden hour, etc.), (5) detailed environment/setting, (6) exact camera specifications (35mm, telephoto, macro, etc.), (7) composition techniques, (8) color palette/mood, and (9) post-processing effects. Use 20-30 tags maximum, prioritizing quality descriptors over quantity. Include 2-3 relevant artist references whose style matches the desired aesthetic. Return ONLY the comma-separated tags without explanations or formatting.",

        "FLUXKontext": "Generate precise Flux Kontext editing instructions using this complete framework: (1) clear action verbs (change, add, remove, replace, transform, modify) with specific target objects and spatial identifiers, (2) detailed modification specs with quantified values (percentages, measurements, exact colors), (3) character consistency protection using 'while maintaining exact same character/facial features/identity' - critical for character preservation, (4) multi-layered preservation clauses for composition/lighting/atmosphere/positioning, (5) specific descriptors avoiding all pronouns - use detailed physical attributes, (6) precise style names with medium characteristics and cultural context, (7) quoted text replacements with typography preservation, (8) semantic relationship maintenance for natural blending, (9) context-aware modifications that understand whole image content. Focus on descriptive language over complex formatting. Ensure edits blend seamlessly with existing content through contextual understanding. Return ONLY the Flux Kontext instruction no more 50 words.",

        "Imagen4": (
            "Craft a vivid, layered image prompt optimized for Imagen 4. "
            "Structure in this precise order: "
            "(1) SUBJECT: Define the main subject(s) with concrete, vivid traits (appearance, pose, expression). "
            "(2) SCENE: Establish environment and context (setting, time of day, background elements). "
            "(3) ATMOSPHERE & LIGHT: Specify lighting—natural or artificial (e.g. golden hour, dramatic side lighting), and mood. "
            "(4) COMPOSITION & TECHNICAL: Describe camera angle, framing, lens effect (e.g. shallow depth of field, wide-angle), perspective and spatial arrangement. "
            "(5) STYLE & QUALITY: Include artistic style and medium (photorealistic, oil painting, illustration), text layout if needed, resolution cues (e.g. 2K, high resolution), and mood-enhancing terms (cinematic, hyper-realistic). "
            "Aim for specificity and clarity; layer in descriptive detail progressively. "
            "Return ONLY the prompt text itself, in a cohesive single paragraph, under 200 tokens."
        ),

        "GeminiNanaBananaEdit": (
            "Convert my editing request into precise Gemini conversational image editing instructions that leverage its mask-free contextual editing capabilities: "
            "(1) CONTEXTUAL REFERENCE: Begin with 'Using the provided image' and identify the specific element to modify using detailed descriptive language rather than spatial coordinates (the blue ceramic vase on the wooden table, the person wearing the red jacket in the center). "
            "(2) EDIT ACTION: Use clear, conversational verbs that specify the transformation (replace with, transform into, add beside, remove while preserving, change the color to, adjust the lighting to make more). "
            "(3) INTEGRATION SPECIFICATION: Describe how the change should blend seamlessly with existing elements, maintaining consistency in lighting, perspective, style, and atmosphere (ensuring the new element matches the existing warm golden hour lighting and rustic kitchen aesthetic). "
            "(4) PRESERVATION DIRECTIVES: Explicitly state what should remain unchanged to protect critical elements (keep everything else exactly the same, preserve the original character's facial features and expression, maintain the architectural details of the background). "
            "(5) STYLE CONTINUITY: Reference the existing visual style and ensure the modification matches (in the same photorealistic style, maintaining the impressionistic brushwork quality, keeping the vintage film photography aesthetic). "
            "(6) RELATIONSHIP CONTEXT: Describe how the edited element should relate to other objects in the scene for natural composition (positioned naturally beside the existing furniture, scaled appropriately for a person of that height, casting realistic shadows on the ground). "
            "Structure as conversational instructions under 75 words that feel like natural directions to an artist who can see and understand the full image context. "
            "Avoid technical jargon and focus on descriptive, intuitive language that leverages Gemini's contextual understanding. "
            "Return ONLY the editing instruction text."
        )
    }

    # Apply template based on prompt_structure parameter
    modified_prompt = prompt
    if prompt_structure != "Custom" and prompt_structure in prompt_templates:
        template = prompt_templates[prompt_structure]
        print(f"Applying {prompt_structure} template")
        modified_prompt = f"{prompt}\n\n{template}"
    else:
        # Fallback to checking if prompt contains a template request
        for template_name, template in prompt_templates.items():
            if template_name.lower() in prompt.lower():
                print(f"Detected {template_name} template request in prompt")
                modified_prompt = f"{prompt}\n\n{template}"
                break

    return modified_prompt

def rgba_to_rgb(image):
    """Convert RGBA image to RGB with white background"""
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
    return image


def tensor_to_pil_image(tensor):
    """Convert tensor to PIL Image with RGBA support"""
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()

    # Handle different channel counts
    if len(image_np.shape) == 2:  # Grayscale
        image_np = np.expand_dims(image_np, axis=-1)
    if image_np.shape[-1] == 1:   # Single channel
        image_np = np.repeat(image_np, 3, axis=-1)

    channels = image_np.shape[-1]
    mode = 'RGBA' if channels == 4 else 'RGB'

    image = Image.fromarray(image_np, mode=mode)
    return rgba_to_rgb(image)


def sample_video_frames(video_tensor, num_samples=6):
    """Sample frames evenly from video tensor"""
    if len(video_tensor.shape) != 4:
        return None

    total_frames = video_tensor.shape[0]
    if total_frames <= num_samples:
        indices = range(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    frames = []
    for idx in indices:
        frame = tensor_to_pil_image(video_tensor[idx])
        frames.append(frame)
    return frames
def get_gemini_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is required")
    return api_key

# ================== API SERVICES ==================


class GeminiLLMAPI:
    def __init__(self):
        self.gemini_api_key = get_gemini_api_key()
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key, transport='rest')

        # Static Gemini model list (manually curated)
        self.available_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro-exp-03-25",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-exp-image-generation",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "imagen-3.0-generate-002",
        ]

    @classmethod
    def INPUT_TYPES(cls):
        # Create an instance to get the models
        instance = cls()
        available_models = instance.available_models

        return {
            "required": {
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "input_type": (["text", "image", "video"], {"default": "text"}),
                "gemini_model": (available_models,),
                "stream": ("BOOLEAN", {"default": False}),
                "structure_output": ("BOOLEAN", {"default": False}),
                "prompt_structure": ([
                    "Custom",
                    "VideoGen",
                    "FLUX.1-dev",
                    "SDXL",
                    "FLUXKontext",
                    "Imagen4",
                    "GeminiNanaBananaEdit"
                ], {"default": "Custom"}),
                "structure_format": ("STRING", {"default": "Return only the prompt text itself. No explanations or formatting.", "multiline": True}),
                "output_format": ([
                    "raw_text",
                    "json"
                ], {"default": "raw_text"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "video": ("IMAGE",),  # Video is represented as a tensor with shape [frames, height, width, channels]
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "AI API/Gemini"

    def generate_content(self, prompt, input_type, gemini_model, stream, structure_output, prompt_structure, structure_format, output_format, api_key="", image1=None, image2=None, image3=None, image4=None, image5=None, video=None):
        if api_key:
            self.gemini_api_key = api_key
            genai.configure(api_key=self.gemini_api_key, transport='rest')
        elif not self.gemini_api_key:
            self.gemini_api_key = get_gemini_api_key()
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key, transport='rest')

        if not self.gemini_api_key:
            return ("Gemini API key missing. Please provide it in the node's api_key input.",)

        try:
            generation_config = {"temperature": 0.7, "top_p": 0.8, "top_k": 40}
            env_model_override = os.getenv("GEMINI_OLLAMA_MODEL")
            current_model = env_model_override or gemini_model
            modified_prompt = apply_prompt_template(prompt, prompt_structure)
            if structure_output:
                print(f"Requesting structured output from {current_model}")
                modified_prompt = f"{modified_prompt}\n\n{structure_format}"
                print(f"Modified prompt with structure format")

            model = genai.GenerativeModel(current_model)

            content = [modified_prompt]

            if input_type == "image":
                # Handle multiple images
                all_images = [image1, image2, image3, image4, image5]
                provided_images = [img for img in all_images if img is not None]

                if provided_images:
                    for img in provided_images:
                        pil_image = tensor_to_pil_image(img)
                        content.append(pil_image)
            elif input_type == "video" and video is not None:
                frames = sample_video_frames(video)
                if frames:
                    content.extend(frames)
                else:
                    return ("Error: Could not extract frames from video",)

            print(f"Sending request to Gemini API with model: {current_model}")

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            response = model.generate_content(
                content,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=stream
            )

            if stream:
                textoutput = "".join([chunk.text for chunk in response if hasattr(chunk, 'text')])
            else:
                if not hasattr(response, 'text'):
                    if hasattr(response, 'prompt_feedback'):
                        return (f"API Error: Content blocked - {response.prompt_feedback}",)
                    else:
                        return (f"API Error: Empty response from Gemini API",)
                textoutput = response.text

            print("Gemini API response received successfully")

            if textoutput.strip():
                clean_text = textoutput.strip()
                if clean_text.startswith("```") and "```" in clean_text[3:]:
                    first_block_end = clean_text.find("```", 3)
                    if first_block_end > 3:
                        language_line_end = clean_text.find("\n", 3)
                        if language_line_end > 3 and language_line_end < first_block_end:
                            clean_text = clean_text[language_line_end+1:first_block_end].strip()
                        else:
                            clean_text = clean_text[3:first_block_end].strip()
                if (clean_text.startswith('"') and clean_text.endswith('"')) or \
                   (clean_text.startswith("'") and clean_text.endswith("'")):
                    clean_text = clean_text[1:-1].strip()
                prefixes_to_remove = ["Prompt:", "PROMPT:", "Generated Prompt:", "Final Prompt:"]
                for prefix in prefixes_to_remove:
                    if clean_text.startswith(prefix):
                        clean_text = clean_text[len(prefix):].strip()
                        break
                if output_format == "json":
                    try:
                        key_name = "prompt"
                        if prompt_structure != "Custom":
                            key_name = f"{prompt_structure.lower().replace('.', '_').replace('-', '_')}_prompt"
                        json_output = json.dumps({key_name: clean_text}, indent=2)
                        print(f"Formatted output as JSON with key: {key_name}")
                        textoutput = json_output
                    except Exception as e:
                        print(f"Error formatting output as JSON: {str(e)}")
                else:
                    textoutput = clean_text
                    print("Returning raw text output")

            return (textoutput,)
        except Exception as e:
            return (f"API Error: {str(e)}",)

# ================== NODE REGISTRATION ==================
NODE_CLASS_MAPPINGS = {
    "GeminiAPI": GeminiLLMAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiAPI": "Gemini API",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
