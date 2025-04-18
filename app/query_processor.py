from typing import List, Dict, Any
import pandas as pd
import numpy as np
import json
from groq import Groq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class QueryProcessor:
    def __init__(self, api_key: str = None):
        self.client = Groq(api_key=api_key)
        
        # Answer generation prompt
        self.answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Context: {context}
            
            Question: {question}
            
            Answer the question based on the context above. If the context doesn't contain enough information to answer the question, say "I cannot answer this question based on the provided context."
            
            For each part of your answer, indicate which source document it came from using [Source: document_name].
            
            Answer:
            """
        )
        
        # Summarization prompt
        self.summary_prompt = PromptTemplate(
            input_variables=["context"],
            template="""
            Please provide a concise summary of the following content. Focus on the key points and main ideas.
            
            Content: {context}
            
            Summary:
            """
        )
        
        # Multiple answers prompt
        self.multiple_answers_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Context: {context}
            
            Question: {question}
            
            Provide multiple possible answers to the question based on the context. Each answer should be clearly numbered and include its source.
            
            If there are conflicting or alternative interpretations, present them as separate answers.
            
            Answers:
            """
        )

    def _call_groq(self, prompt: str) -> str:
        """Make a call to Groq API"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="mixtral-8x7b-32768",  # Using Mixtral model
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            return "Error generating response. Please try again."

    def process_query(self, query: str, relevant_chunks: List[Dict[str, Any]], mode: str = "single") -> Dict[str, Any]:
        """Process a query and generate an answer"""
        # Combine relevant chunks into context
        context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        
        # Check if any chunks are from structured data
        structured_data = self._extract_structured_data(relevant_chunks)
        
        if structured_data:
            # Try to answer using structured data first
            answer = self._process_structured_query(query, structured_data)
            if answer:
                return {
                    "answer": answer,
                    "sources": [chunk["metadata"] for chunk in relevant_chunks],
                    "type": "structured"
                }
        
        # Process based on mode
        if mode == "summary":
            prompt = self.summary_prompt.format(context=context)
            answer = self._call_groq(prompt)
        elif mode == "multiple":
            prompt = self.multiple_answers_prompt.format(context=context, question=query)
            answer = self._call_groq(prompt)
        else:
            prompt = self.answer_prompt.format(context=context, question=query)
            answer = self._call_groq(prompt)
        
        # Extract references from answer
        references = self._extract_references(answer)
        
        return {
            "answer": answer,
            "sources": [chunk["metadata"] for chunk in relevant_chunks],
            "references": references,
            "type": "text"
        }

    def _extract_references(self, answer: str) -> List[Dict[str, Any]]:
        """Extract references from the answer text"""
        references = []
        lines = answer.split('\n')
        
        for line in lines:
            if '[Source:' in line:
                source_start = line.find('[Source:')
                source_end = line.find(']', source_start)
                if source_start != -1 and source_end != -1:
                    source = line[source_start + 8:source_end].strip()
                    references.append({
                        "text": line[:source_start].strip(),
                        "source": source
                    })
        
        return references

    def _extract_structured_data(self, chunks: List[Dict[str, Any]]) -> List[pd.DataFrame]:
        """Extract structured data from chunks"""
        dfs = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            if metadata.get("type") in ["excel", "csv"]:
                try:
                    df = pd.read_json(chunk["content"])
                    dfs.append(df)
                except:
                    continue
        return dfs

    def _process_structured_query(self, query: str, dfs: List[pd.DataFrame]) -> str:
        """Process queries against structured data"""
        try:
            # Simple pattern matching for common query types
            if "sum" in query.lower() or "total" in query.lower():
                for df in dfs:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        return f"The sum is {df[numeric_cols[0]].sum()}"
            
            if "average" in query.lower() or "mean" in query.lower():
                for df in dfs:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        return f"The average is {df[numeric_cols[0]].mean()}"
            
            return None
        except:
            return None 