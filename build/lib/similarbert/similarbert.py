import re
from itertools import combinations
from collections import Counter
import pandas as pd
from nltk import ngrams
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

class SimilarBERT:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        
        self.model = SentenceTransformer(model_name)
        
            
    def fit(self, docs):
        
        embeddings = self.model.encode(docs)
        return embeddings, docs
        
            
    def transform(self, embeddings, docs, custom_topics=None, threshold=0.4, min_community_size=10, batch_size=128):
        
        if custom_topics is None:
            topics = self.community_detection(embeddings, docs, threshold, min_community_size, batch_size)
        else:
            topics = self.custom_topic_clusters_df(docs, custom_topics, threshold)
        return topics
        
            

    def fit_transform(self, docs, custom_topics=None):
        
        embeddings, docs = self.fit(docs)
        if custom_topics is None:
            topics = self.community_detection(embeddings, docs, threshold=0.75, min_community_size=10, batch_size=128)
        else:
            topics = self.custom_topic_clusters_df(docs, custom_topics, threshold=0.7)
        return topics
        
            
    def community_detection(self, embeddings, document_list, threshold, min_community_size, batch_size):
        
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        threshold = torch.tensor(threshold, device=embeddings.device)

        extracted_communities = []

        min_community_size = min(min_community_size, len(embeddings))
        sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

        for start_idx in range(0, len(embeddings), batch_size):
            cos_scores = F.cosine_similarity(embeddings[start_idx:start_idx + batch_size].unsqueeze(1), embeddings.unsqueeze(0), dim=2)

            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    new_cluster = []

                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                        sort_max_size = min(2 * sort_max_size, len(embeddings))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    for idx, val in zip(top_idx_large.tolist(), top_val_large):
                        if val < threshold:
                            break

                        new_cluster.append(idx)

                    extracted_communities.append(new_cluster)

        extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

        unique_communities = []
        extracted_ids = set()

        for cluster_id, community in enumerate(extracted_communities):
            community = sorted(community)
            non_overlapped_community = []
            for idx in community:
                if idx not in extracted_ids:
                    non_overlapped_community.append(idx)

            if len(non_overlapped_community) >= min_community_size:
                unique_communities.append(non_overlapped_community)
                extracted_ids.update(non_overlapped_community)

        outliers = [idx for idx in range(len(embeddings)) if idx not in extracted_ids]
        if len(outliers) > 0:
            unique_communities.append(outliers)

        community_data = []
        for i, community in enumerate(unique_communities):
            community_embeddings = [embeddings[idx].tolist() for idx in community]
            community_documents = [document_list[sentence_id] for sentence_id in community]

            topic_heading = self.find_ngrams(community_documents)
            community_data.append({
                "Topic No.": i + 1,
                "Topic Heading": topic_heading,
                "Number of Documents": len(community),
                "Embeddings": community_embeddings,
                "Documents": community_documents
            })

        community_df = pd.DataFrame(community_data)
        community_df["Unique_Communities"] = unique_communities
        return community_df
    
        

    def custom_topic_clusters_df(self, docs, custom_topics, threshold):
        
        docs_embeddings = self.fit(docs)
        topics_embeddings = self.fit(custom_topics)
        similarity_matrix = cosine_similarity(topics_embeddings, docs_embeddings)

        num_custom_topics = len(custom_topics)
        num_docs = len(docs)

        custom_topic_clusters = [[] for _ in range(num_custom_topics)]
        custom_topic_data = []

        for i in range(num_custom_topics):
            custom_topic = custom_topics[i]

            unique_communities = set()
            topic_embeddings = []

            for j in range(num_docs):
                similarity_score = similarity_matrix[i][j]
                if similarity_score >= threshold:
                    custom_topic_clusters[i].append(docs[j])
                    unique_communities.add(j)
                    topic_embeddings.append(docs_embeddings[j])

            row_data = {
                "Topic No.": i + 1,
                "Number of Documents": len(custom_topic_clusters[i]),
                "Embeddings": topic_embeddings,
                "Documents": custom_topic_clusters[i],
                "Unique_Communities": list(unique_communities)
            }

            custom_topic_data.append(row_data)

        custom_topic_df = pd.DataFrame(custom_topic_data)
        return custom_topic_df
    
        

    def preprocess(self, sentences, remove_symbols=True, remove_stopwords=True):
        
        pattern = re.compile(r'[\W_]+')
        processed_sentences = []

        for sentence in sentences:
            if remove_symbols:
                sentence = re.sub(pattern, ' ', sentence)

            if remove_stopwords:
                stop_words = set(stopwords.words('english'))
                words = word_tokenize(sentence)
                words = [word for word in words if word.lower() not in stop_words]
                sentence = ' '.join(words)

            processed_sentences.append(sentence)

        return processed_sentences
    
      

    def find_ngrams(self, sentences, n=2, method='frequency', top_n=5):
        
        if method == 'frequency':
            cleaned_sentences = self.preprocess(sentences)
            ngram_phrases = []
            for sentence in cleaned_sentences:
                words = word_tokenize(sentence)
                ngram_sentence = list(ngrams(words, n))
                ngram_phrases.extend([' '.join(gram) for gram in ngram_sentence])

            ngram_counts = Counter(ngram_phrases)
            most_frequent_ngrams = ngram_counts.most_common(top_n) if top_n else ngram_counts.most_common()
            return most_frequent_ngrams

        elif method == 'tfidf':
            cleaned_sentences = self.preprocess(sentences, remove_symbols=False)
            tfidf_vectorizer = TfidfVectorizer(ngram_range=(n, n))
            tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_sentences)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            ngram_tfidf_dict = dict(zip(feature_names, tfidf_scores))
            sorted_ngram_tfidf = sorted(ngram_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
            top_ngram_phrases = sorted_ngram_tfidf[:top_n] if top_n else sorted_ngram_tfidf
            return top_ngram_phrases

        else:
            raise ValueError("Invalid method. Choose 'frequency' or 'tfidf'.")
    
  

            
            
    def get_topic_list(self, topics=None):
        
        topic_list = topics.copy() 
        columns_to_drop = ['Unique_Communities', 'Embeddings']
        topic_list.drop(columns=columns_to_drop, inplace=True)
        topic_list.reset_index(drop=True, inplace=True)
        return topic_list
    

    def top_n_topics(self, topics=None, top_n=5):
        
        topics = topics.nlargest(top_n, 'Number of Documents')
        for _, row in topics.iterrows():
            print(f"Topic {row['Topic No.']}:")
            print("-" * 97)  
            print('Topic Heading:')
            print("-" * 97)  
            for heading in row['Topic Heading']:
                print(f"{heading[0]}: {heading[1]}")
            print("-" * 97) 
            print('Documents:')
            print("-" * 97)  
            for i, document in enumerate(row['Documents'], start=1):
                print(f"Document {i}:\n{document}")
            print("\n")
    
        

    def bottom_n_topics(self, topics=None, bottom_n=5):

        bottom_topics = topics.nsmallest(bottom_n, 'Number of Documents')
        for _, row in bottom_topics.iterrows():
            print(f"Topic {row['Topic No.']}:")
            print("-" * 97)
            print('Topic Heading:')
            print("-" * 97)
            for heading in row['Topic Heading']:
                print(f"{heading[0]}: {heading[1]}")
            print("-" * 97)
            print('Documents:')
            print("-" * 97)
            for i, document in enumerate(row['Documents'], start=1):
                print(f"Document {i}:\n{document}")
            print("\n")
        
           
    def select_topics(self, topics=None, topics_to_return=[1, 2, 3, 4, 5]):

        selected_topics = topics[topics["Topic No."].isin(topics_to_return)]
        for _, row in selected_topics.iterrows():
            print(f"Topic {row['Topic No.']}:")
            print("-" * 97)
            print('Topic Heading:')
            print("-" * 97)
            for heading in row['Topic Heading']:
                print(f"{heading[0]}: {heading[1]}")
            print("-" * 97)
            print('Documents:')
            print("-" * 97)
            for i, document in enumerate(row['Documents'], start=1):
                print(f"Document {i}:\n{document}")
            print("\n")
        
           
    def single_topic(self, topics=None, topic_number=1):
        
        selected_topic = topics[topics["Topic No."] == topic_number]
        if not selected_topic.empty:
            for _, row in selected_topic.iterrows():
                print(f"Topic {row['Topic No.']}:")
                print("-" * 97)
                print('Topic Heading:')
                print("-" * 97)
                for heading in row['Topic Heading']:
                    print(f"{heading[0]}: {heading[1]}")
                print("-" * 97) 
                print('Documents:')
                print("-" * 97)
                for i, document in enumerate(row['Documents'], start=1):
                    print(f"Document {i}:\n{document}")
        
            
    def list_of_docs(self, topics=None):

        document_topic_data = []
        for i, topic in topics.iterrows():
            topic_no = topic['Topic No.']
            documents = topic['Documents']
            for document in documents:
                document_topic_data.append({'Topic': topic_no, 'Document': document})
        list_of_docs = pd.DataFrame(document_topic_data)

        return list_of_docs
        
    #VISUALIZATION

    def vis_corpus_wordcloud(self, topic_list=None, width=800, height=400, background_color='white', colormap='viridis'):
        if topic_list is not None:
            corpus = ""
            for _, row in topic_list.iterrows():
                for document in row['Documents']:
                    corpus += document + " "

            wordcloud = WordCloud(width=width, height=height, background_color=background_color, colormap=colormap).generate(corpus)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
        
            
    def vis_wordcloud_top_n_topics(self, topic_list=None, top_n=5, width=1200, height=800, background_color='white', colormap='viridis', title_font_size=20):

        if topic_list is not None:
            num_rows = (top_n + 2) // 3
            subplot_width = 8
            subplot_height = 6
            figsize_width = subplot_width * 3
            figsize_height = subplot_height * num_rows

            fig, axes = plt.subplots(num_rows, 3, figsize=(figsize_width, figsize_height))
            fig.subplots_adjust(hspace=0.0, wspace=0.0)

            for i in range(top_n):
                row = topic_list.iloc[i]
                topic_docs = " ".join(row['Documents'])
                topic_docs = self.preprocess([topic_docs], remove_symbols=True, remove_stopwords=True)[0]

                wordcloud = WordCloud(width=width, height=height, background_color=background_color, colormap=colormap).generate(topic_docs)

                ax = axes[i // 3, i % 3]
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f"Topic {row['Topic No.']}", fontsize=title_font_size)  # Set title font size
                ax.axis('off')

            for i in range(top_n, num_rows * 3):
                axes[i // 3, i % 3].axis('off')

            plt.show()
    
    def vis_wordcloud_select_topics(self, topic_list=None, topics_to_generate=None, width=1200, height=800, background_color='white', colormap='viridis', title_font_size=20):
        
        if topic_list is not None and topics_to_generate is not None:
            num_topics = len(topics_to_generate)
            num_rows = (num_topics + 2) // 3  
            
            subplot_width = 8
            subplot_height = 6
            figsize_width = subplot_width * 3
            figsize_height = subplot_height * num_rows

            fig, axes = plt.subplots(num_rows, 3, figsize=(figsize_width, figsize_height))
            fig.subplots_adjust(hspace=0.0, wspace=0.0) 

            for i, topic_number in enumerate(topics_to_generate):
                row = topic_list[topic_list['Topic No.'] == topic_number].iloc[0]

                topic_docs = " ".join(row['Documents']) 
                topic_docs = self.preprocess([topic_docs], remove_symbols=True, remove_stopwords=True)[0]


                wordcloud = WordCloud(width=width, height=height, background_color=background_color, colormap=colormap).generate(topic_docs)

                ax = axes[i // 3, i % 3]
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f"Topic {topic_number}", fontsize=title_font_size)  # Set title font size
                ax.axis('off')

            for i in range(num_topics, num_rows * 3):
                axes[i // 3, i % 3].axis('off')

            # Show the plot with all word clouds
            plt.show()
        else:
            print("Please provide valid topic_list and topics_to_generate.")
    
    def vis_topics_barchart(self, topic_list=None, topic_numbers=[1], n=2, method='frequency', top_n=5, color=None):
        
        if topic_list is not None:
            num_topics = len(topic_numbers)
            num_cols = 2  
            num_rows = (num_topics + 1) // num_cols 

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))
            fig.subplots_adjust(hspace=0.4)

            for i, topic_number in enumerate(topic_numbers):
                row_idx = i // num_cols
                col_idx = i % num_cols

                documents = topic_list[topic_list['Topic No.'] == topic_number]['Documents'].iloc[0]

                ngram_phrases = self.find_ngrams(documents, n=n, method=method, top_n=top_n)

                phrases = [phrase[0] for phrase in ngram_phrases]
                frequencies = [phrase[1] for phrase in ngram_phrases]

                ax = axes[row_idx, col_idx]
                ax.barh(phrases, frequencies, color=color)  
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Phrases')
                ax.set_title(f'Topic {topic_number} (Method: {method})')
                ax.invert_yaxis() 

            for i in range(num_topics, num_rows * num_cols):
                row_idx = i // num_cols
                col_idx = i % num_cols
                fig.delaxes(axes[row_idx, col_idx])

            plt.tight_layout()
            plt.show()
        else:
            print("Please provide a valid topic_list.")
        
              
            
    # TOPIC SIMILARITY

    def cal_similarity_score(self, embeddings):
        
        similarity_matrix = cosine_similarity(embeddings)

        num_sentences = len(embeddings)
        avg_coherence = 0.0
        pair_count = 0

        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                avg_coherence += similarity_matrix[i][j]
                pair_count += 1

        if pair_count == 0:
            return 0.0

        avg_coherence /= pair_count
        return avg_coherence
        
            
    def topic_similarity(self, topics=None, top_n=None, sort=True, sort_by="Topic Similarity"):
        
        if topics is not None:
            topic_data = []

            for i, row in topics.iterrows():
                community_embeddings = row["Embeddings"]
                community_embeddings = torch.tensor(community_embeddings)
                community_similarity = self.cal_similarity_score(community_embeddings)

                topic_data.append({
                    "Topic No.": i,
                    "Number of Documents": row["Number of Documents"],
                    "Documents": row['Documents'],
                    "Topic Similarity": community_similarity
                })

            topic_df = pd.DataFrame(topic_data)

            if sort or sort_by is not None:
                topic_df.sort_values(by=sort_by, ascending=False, inplace=True)

            top_n_communities = topic_df.head(top_n)

            return top_n_communities
        else:
            print("Please provide valid topics.")
            return None
    
    def vis_topic_similarity_chart(self, topics=None, chart_type='bar', top_n=None, sort_by=None, color='g'):
        
        if topics is not None:
            if sort_by:
                topic_df = self.topic_similarity(topics=topics, sort_by=sort_by, top_n=top_n)
            else:
                topic_df = self.topic_similarity(topics=topics, sort_by="Topic No.", top_n=top_n)

            if chart_type not in ['bar', 'line', 'scatter']:
                print("Invalid chart type. Please choose 'bar', 'line', or 'scatter'.")
                return

            topic_numbers = topic_df["Topic No."]
            similarity_scores = topic_df["Topic Similarity"]

            plt.figure(figsize=(10, 6))

            if chart_type == 'bar':
                plt.bar(topic_numbers, similarity_scores, align='center', alpha=0.7, color=color)
                plt.xlabel("Topic Number")
                plt.ylabel("Topic Similarity")
                plt.title("Topic Similarity Bar Chart")
            elif chart_type == 'line':
                plt.plot(topic_numbers, similarity_scores, marker='o', linestyle='-', color=color)
                plt.xlabel("Topic Number")
                plt.ylabel("Topic Similarity")
                plt.title("Topic Similarity Line Chart")
            elif chart_type == 'scatter':
                plt.scatter(topic_numbers, similarity_scores, color=color)
                plt.xlabel("Topic Number")
                plt.ylabel("Topic Similarity")
                plt.title("Topic Similarity Scatter Chart")

            plt.grid(True)
            plt.show()
        else:
            print("Please provide valid topics.")
    
    def vis_topic_similarity_heatmap(self, topics=None, topic_no=None):
        
        if topics is not None and topic_no is not None:
            embeddings = topics['Embeddings'].loc[topic_no]

            similarity_matrix = cosine_similarity(embeddings)

            plt.figure(figsize=(8, 6))
            sns.heatmap(similarity_matrix, annot=True, fmt=".2f")
            plt.title(f"Topic {topic_no} Similarity Heatmap")
            plt.show()
        else:
            print("Please provide valid topics and a valid topic number.")
   
            
            
            
    #TOPIC DISCIMILARITY
            
    def cal_dissimilarity_score(self, embeddings):
        
        similarity_matrix = 1 - cosine_similarity(embeddings)

        num_communities = len(embeddings)
        avg_dissimilarity = 0.0
        pair_count = 0

        for i in range(num_communities):
            for j in range(i + 1, num_communities):
                avg_dissimilarity += similarity_matrix[i][j]
                pair_count += 1

        if pair_count == 0:
            return 0.0

        avg_dissimilarity /= pair_count
        return avg_dissimilarity
  

    def topic_dissimilarity(self, topics=None, top_n=None, sort=True, sort_by="Topic Dissimilarity"):
        
        topic_data = []

        for i, row in topics.iterrows():
            community_embeddings = row["Embeddings"]
            community_embeddings = torch.tensor(community_embeddings)

            community_dissimilarity = self.cal_dissimilarity_score(community_embeddings)
            topic_data.append({
                "Topic No.": i,
                "Number of Documents": row["Number of Documents"],
                "Documents": row['Documents'],
                "Topic Dissimilarity": community_dissimilarity
            })

        topic_df = pd.DataFrame(topic_data)

        if sort or sort_by is not None:
            topic_df.sort_values(by=sort_by, ascending=False, inplace=True)

        top_n_communities = topic_df.head(top_n)

        return top_n_communities
    
        
    def vis_topic_dissimilarity_chart(self, topics=None, chart_type='bar', top_n=None, sort_by=None, color='g'):
        
        if sort_by:
            topic_df = self.topic_dissimilarity(topics=topics, sort_by=sort_by, top_n=top_n)
        else:
            topic_df = self.topic_dissimilarity(topics=topics, sort_by="Topic No.", top_n=top_n)

        if chart_type not in ['bar', 'line', 'scatter']:
            print("Invalid chart type. Please choose 'bar', 'line', or 'scatter'.")
            return

        topic_numbers = topic_df["Topic No."]
        dissimilarity_scores = topic_df["Topic Dissimilarity"]

        plt.figure(figsize=(10, 6))

        if chart_type == 'bar':
            plt.bar(topic_numbers, dissimilarity_scores, align='center', alpha=0.7, color=color)
            plt.xlabel("Topic Number")
            plt.ylabel("Topic Dissimilarity")
            plt.title("Topic Dissimilarity Bar Chart")
        elif chart_type == 'line':
            plt.plot(topic_numbers, dissimilarity_scores, marker='o', linestyle='-', color=color)
            plt.xlabel("Topic Number")
            plt.ylabel("Topic Dissimilarity")
            plt.title("Topic Dissimilarity Line Chart")
        elif chart_type == 'scatter':
            plt.scatter(topic_numbers, dissimilarity_scores, color=color)
            plt.xlabel("Topic Number")
            plt.ylabel("Topic Dissimilarity")
            plt.title("Topic Dissimilarity Scatter Chart")

        plt.grid(True)
        plt.show()
    
    def topic_dissimilarity_heatmap(self, topics=None, topic_no=None):
        
        if topics is not None and topic_no is not None:
            embeddings = topics['Embeddings'].loc[topic_no]

            dissimilarity_matrix = 1 - cosine_similarity(embeddings)

            plt.figure(figsize=(8, 6))
            sns.heatmap(dissimilarity_matrix, annot=True, fmt=".2f")
            plt.title(f"Topic {topic_no} Dissimilarity Heatmap")
            plt.show()
        else:
            print("Please provide valid topics and a valid topic number.")
    
            
    def topic_similarity_and_dissimilarity_chart(self, topics=None, chart_type='bar', top_n=None, sort_by=None, color_sim='g', color_dissim='r'):
        
        if sort_by:
            topic_sim_df = self.topic_similarity(topics=topics, sort_by=sort_by, top_n=top_n)
            topic_dissim_df = self.topic_dissimilarity(topics=topics, sort_by=sort_by, top_n=top_n)
        else:
            topic_sim_df = self.topic_similarity(topics=topics, sort_by="Topic No.", top_n=top_n)
            topic_dissim_df = self.topic_dissimilarity(topics=topics, sort_by="Topic No.", top_n=top_n)

        if chart_type not in ['bar', 'line', 'scatter']:
            print("Invalid chart type. Please choose 'bar', 'line', or 'scatter'.")
            return

        topic_numbers_sim = topic_sim_df["Topic No."]
        similarity_scores = topic_sim_df["Topic Similarity"]

        topic_numbers_dissim = topic_dissim_df["Topic No."]
        dissimilarity_scores = topic_dissim_df["Topic Dissimilarity"]

        plt.figure(figsize=(10, 6))

        if chart_type == 'bar':
            plt.bar(topic_numbers_sim, similarity_scores, align='center', alpha=0.7, color=color_sim, label="Similarity")
            plt.bar(topic_numbers_dissim, dissimilarity_scores, align='edge', alpha=0.7, color=color_dissim, label="Dissimilarity")
            plt.xlabel("Topic Number")
            plt.ylabel("Topic Score")
            plt.title("Topic Similarity and Dissimilarity Bar Chart")
        elif chart_type == 'line':
            plt.plot(topic_numbers_sim, similarity_scores, marker='o', linestyle='-', color=color_sim, label="Similarity")
            plt.plot(topic_numbers_dissim, dissimilarity_scores, marker='o', linestyle='-', color=color_dissim, label="Dissimilarity")
            plt.xlabel("Topic Number")
            plt.ylabel("Topic Score")
            plt.title("Topic Similarity and Dissimilarity Line Chart")
        elif chart_type == 'scatter':
            plt.scatter(topic_numbers_sim, similarity_scores, color=color_sim, label="Similarity")
            plt.scatter(topic_numbers_dissim, dissimilarity_scores, color=color_dissim, label="Dissimilarity")
            plt.xlabel("Topic Number")
            plt.ylabel("Topic Score")
            plt.title("Topic Similarity and Dissimilarity Scatter Chart")

        plt.legend()
        plt.grid(True)
        plt.show()
    
 

    