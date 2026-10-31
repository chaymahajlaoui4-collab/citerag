# CiteRAG

Assistant RAG qui répond à des questions sur la documentation Camunda 8 et **cite ses sources**. Il refuse de répondre quand l'information n'est pas dans les documents.

## Fonctionnalités
- Ingestion de fichiers Markdown/PDF/texte, découpage en chunks, embeddings multilingues (`multilingual-e5-small`), base vectorielle Chroma
- Réponses générées par Groq (`openai/gpt-oss-120b`) avec citations `[fichier, p.X]`
- API FastAPI (`POST /ask`) et interface Streamlit
- Couche agent (function calling) : le modèle décide quand chercher, jusqu'à 3 recherches
- Script d'évaluation (Hit@1, Hit@5, taux de refus hors corpus)
- Docker / docker-compose

## Architecture
Documents -> chunking -> embeddings -> Chroma -> recherche top-5 -> LLM (Groq) -> réponse + sources

## Installation
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # puis mettre sa clé GROQ_API_KEY
# placer des .md / .pdf / .txt dans data/
python3 ingest.py
python3 ask.py "Qu'est-ce qu'un cluster dans Camunda ?"
```
API : `uvicorn api:app --reload` | Interface : `streamlit run app.py`
Docker : `docker compose up --build` (API sur :8000, interface sur :8501)

## Évaluation
Corpus : 35 fichiers de `docs/components/concepts` (Camunda), 224 chunks à 300 mots.
Jeu de test : 18 questions avec réponse + 5 hors corpus.

| CHUNK_WORDS | Chunks | Hit@1 | Hit@5 | Refus hors corpus |
|---|---|---|---|---|
| 150 | 531 | 78 % (14/18) | 100 % | 4/5 |
| 300 | 224 | 83 % (15/18) | 100 % | 5/5 |
| 500 | 132 | 83 % (15/18) | 100 % | 4/5 |

Conclusion : sur ce jeu de test, la taille des chunks n'a pas d'effet mesurable sur le retrieval (écart d'une question). Je garde 300.

## Problème trouvé et corrigé
La question hors corpus « Comment installer Camunda sur Kubernetes avec Helm ? » a produit une fausse procédure avec de fausses citations : les passages ne contenaient que des liens, et le modèle a complété avec ses connaissances générales. Correctif : prompt plus strict (un lien n'est pas une réponse) et phrase de refus fixe. Refus passé de 1/2 à 2/2 sur le premier jeu de test.

## Limites connues
- Jeu de test petit (23 questions) : un écart d'une question n'est pas significatif.
- Fichiers voisins parfois confondus (migration/modification, suspension/deletion) : Hit@1 à 83 %.
- Le refus dépend du LLM et n'est pas parfaitement stable d'une exécution à l'autre.
- L'agent consomme beaucoup plus de tokens (historique renvoyé à chaque tour) ; son évaluation complète n'a pas pu être menée sur le palier gratuit Groq (limite de 200 000 tokens par jour).
- Corpus limité à un sous-ensemble de la documentation.

## Pistes
Reranker, modèle d'embeddings `bge-m3`, plus de questions de test, seuil de refus calibré.
