from googletrans import Translator

translator = Translator()

def translate_to_english(text, src_lang=None):
    '''
    Translates input text to English. Returns (translated_text, detected_source_language).
    '''
    result = translator.translate(text, src='auto' if src_lang is None else src_lang, dest='en')
    return result.text, result.src 