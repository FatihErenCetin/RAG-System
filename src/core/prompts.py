"""LLM prompt template'leri — tek merkez."""

QA_SYSTEM_PROMPT = """Sen yalnızca sana verilen dokümanlardan bilgi kullanarak cevap veren bir asistansın.
Eğer cevap dokümanlarda yoksa, kesinlikle uydurma; "Bu soru için yeterli bilgi bulunamadı" de.
Cevabını soru hangi dildeyse o dilde ver (Türkçe soru → Türkçe cevap, English question → English answer)."""


def build_qa_prompt(question: str, context_blocks: list[str]) -> str:
    """Q&A için tam prompt oluştur."""
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(Doküman sağlanmadı)"
    return f"""{QA_SYSTEM_PROMPT}

Dokümanlar:
{context_text}

Soru: {question}

Cevap:"""


def build_summarization_prompt(document_name: str, content: str) -> str:
    """Doküman özeti için prompt oluştur."""
    return f"""Aşağıdaki doküman içeriğinin kapsamlı ama öz bir özetini çıkar.
Ana temaları, önemli bulgularını ve sonuçları içersin.
Eğer doküman Türkçe ise özeti Türkçe, İngilizce ise İngilizce yaz.

Doküman adı: {document_name}
İçerik:
{content}

Özet:"""
