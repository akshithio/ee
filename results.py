from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors, fasttext
from gensim.scripts.glove2word2vec import glove2word2vec

# legacy code since it is necessary only once.

glove_orig_50vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/50-vec/50m/vectors.txt"
glove_orig_50vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/50-vec/100m/vectors.txt"
glove_orig_50vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/50-vec/150m/vectors.txt"

glove_orig_100vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/100-vec/50m/vectors.txt"
glove_orig_100vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/100-vec/100m/vectors.txt"
glove_orig_100vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/100-vec/150m/vectors.txt"

glove_orig_150vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/150-vec/50m/vectors.txt"
glove_orig_150vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/150-vec/100m/vectors.txt"
glove_orig_150vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove/150-vec/150m/vectors.txt"


glove_50vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/50-vec/50m/vectors.txt"
glove_50vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/50-vec/100m/vectors.txt"
glove_50vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/50-vec/150m/vectors.txt"

glove_100vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/100-vec/50m/vectors.txt"
glove_100vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/100-vec/100m/vectors.txt"
glove_100vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/100-vec/150m/vectors.txt"

glove_150vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/150-vec/50m/vectors.txt"
glove_150vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/150-vec/100m/vectors.txt"
glove_150vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/glove-conversion/150-vec/150m/vectors.txt"


fastText_50vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/50-vec/50m/vectors.bin"
fastText_50vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/50-vec/100m/vectors.bin"
fastText_50vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/50-vec/150m/vectors.bin"

fastText_100vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/100-vec/50m/vectors.bin"
fastText_100vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/100-vec/100m/vectors.bin"
fastText_100vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/100-vec/150m/vectors.bin"

fastText_150vec_50m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/150-vec/50m/vectors.bin"
fastText_150vec_100m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/150-vec/100m/vectors.bin"
fastText_150vec_150m = "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/new-embeddings/fastText/150-vec/150m/vectors.bin"


# legacy code since it is necessary only once.

_ = glove2word2vec(
    glove_orig_50vec_50m, glove_50vec_50m)
_ = glove2word2vec(
    glove_orig_50vec_100m, glove_50vec_100m)
_ = glove2word2vec(
    glove_orig_50vec_150m, glove_50vec_150m)

_ = glove2word2vec(
    glove_orig_100vec_50m, glove_100vec_50m)
_ = glove2word2vec(
    glove_orig_100vec_100m, glove_100vec_100m)
_ = glove2word2vec(
    glove_orig_100vec_150m, glove_100vec_150m)

_ = glove2word2vec(
    glove_orig_150vec_50m, glove_150vec_50m)
_ = glove2word2vec(
    glove_orig_150vec_100m, glove_150vec_100m)
_ = glove2word2vec(
    glove_orig_150vec_150m, glove_150vec_150m)


glove_model_50vec_50m = KeyedVectors.load_word2vec_format(
    glove_50vec_50m)
glove_model_50vec_100m = KeyedVectors.load_word2vec_format(
    glove_50vec_100m)
glove_model_50vec_150m = KeyedVectors.load_word2vec_format(
    glove_50vec_150m)

glove_model_100vec_50m = KeyedVectors.load_word2vec_format(
    glove_100vec_50m)
glove_model_100vec_100m = KeyedVectors.load_word2vec_format(
    glove_100vec_100m)
glove_model_100vec_150m = KeyedVectors.load_word2vec_format(
    glove_100vec_150m)

glove_model_150vec_50m = KeyedVectors.load_word2vec_format(
    glove_150vec_50m)
glove_model_150vec_100m = KeyedVectors.load_word2vec_format(
    glove_150vec_100m)
glove_model_150vec_150m = KeyedVectors.load_word2vec_format(
    glove_150vec_150m)


fastText_model_50vec_50m = fasttext.load_facebook_vectors(
    fastText_50vec_50m)
fastText_model_50vec_100m = fasttext.load_facebook_vectors(
    fastText_50vec_100m)
fastText_model_50vec_150m = fasttext.load_facebook_vectors(
    fastText_50vec_150m)

fastText_model_100vec_50m = fasttext.load_facebook_vectors(
    fastText_100vec_50m)
fastText_model_100vec_100m = fasttext.load_facebook_vectors(
    fastText_100vec_100m)
fastText_model_100vec_150m = fasttext.load_facebook_vectors(
    fastText_100vec_150m)

fastText_model_150vec_50m = fasttext.load_facebook_vectors(
    fastText_150vec_50m)
fastText_model_150vec_100m = fasttext.load_facebook_vectors(
    fastText_150vec_100m)
fastText_model_150vec_150m = fasttext.load_facebook_vectors(
    fastText_150vec_150m)

# testing = glove_model_150vec_150m.evaluate_word_pairs(datapath(
#     "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/wordsim353/combined.csv"))


fastText_corr_50vec_50m_simlex = fastText_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
fastText_corr_50vec_100m_simlex = fastText_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
fastText_corr_50vec_150m_simlex = fastText_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
fastText_corr_100vec_50m_simlex = fastText_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
fastText_corr_100vec_100m_simlex = fastText_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
fastText_corr_100vec_150m_simlex = fastText_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
fastText_corr_150vec_50m_simlex = fastText_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
fastText_corr_150vec_100m_simlex = fastText_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
fastText_corr_150vec_150m_simlex = fastText_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))

glove_corr_50vec_50m_simlex = glove_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
glove_corr_50vec_100m_simlex = glove_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
glove_corr_50vec_150m_simlex = glove_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
glove_corr_100vec_50m_simlex = glove_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
glove_corr_100vec_100m_simlex = glove_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
glove_corr_100vec_150m_simlex = glove_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
glove_corr_150vec_50m_simlex = glove_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
glove_corr_150vec_100m_simlex = glove_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))
glove_corr_150vec_150m_simlex = glove_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/simlex.tsv"))


fastText_corr_50vec_50m_wordsim = fastText_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
fastText_corr_50vec_100m_wordsim = fastText_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
fastText_corr_50vec_150m_wordsim = fastText_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
fastText_corr_100vec_50m_wordsim = fastText_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
fastText_corr_100vec_100m_wordsim = fastText_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
fastText_corr_100vec_150m_wordsim = fastText_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
fastText_corr_150vec_50m_wordsim = fastText_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
fastText_corr_150vec_100m_wordsim = fastText_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
fastText_corr_150vec_150m_wordsim = fastText_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))

glove_corr_50vec_50m_wordsim = glove_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
glove_corr_50vec_100m_wordsim = glove_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
glove_corr_50vec_150m_wordsim = glove_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
glove_corr_100vec_50m_wordsim = glove_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
glove_corr_100vec_100m_wordsim = glove_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
glove_corr_100vec_150m_wordsim = glove_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
glove_corr_150vec_50m_wordsim = glove_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
glove_corr_150vec_100m_wordsim = glove_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))
glove_corr_150vec_150m_wordsim = glove_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/wordsim.tsv"))


fastText_corr_50vec_50m_men = fastText_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
fastText_corr_50vec_100m_men = fastText_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
fastText_corr_50vec_150m_men = fastText_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
fastText_corr_100vec_50m_men = fastText_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
fastText_corr_100vec_100m_men = fastText_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
fastText_corr_100vec_150m_men = fastText_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
fastText_corr_150vec_50m_men = fastText_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
fastText_corr_150vec_100m_men = fastText_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
fastText_corr_150vec_150m_men = fastText_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))

glove_corr_50vec_50m_men = glove_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
glove_corr_50vec_100m_men = glove_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
glove_corr_50vec_150m_men = glove_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
glove_corr_100vec_50m_men = glove_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
glove_corr_100vec_100m_men = glove_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
glove_corr_100vec_150m_men = glove_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
glove_corr_150vec_50m_men = glove_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
glove_corr_150vec_100m_men = glove_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))
glove_corr_150vec_150m_men = glove_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men.tsv"))


fastText_corr_50vec_50m_men_anno1 = fastText_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
fastText_corr_50vec_100m_men_anno1 = fastText_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
fastText_corr_50vec_150m_men_anno1 = fastText_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
fastText_corr_100vec_50m_men_anno1 = fastText_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
fastText_corr_100vec_100m_men_anno1 = fastText_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
fastText_corr_100vec_150m_men_anno1 = fastText_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
fastText_corr_150vec_50m_men_anno1 = fastText_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
fastText_corr_150vec_100m_men_anno1 = fastText_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
fastText_corr_150vec_150m_men_anno1 = fastText_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))

glove_corr_50vec_50m_men_anno1 = glove_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
glove_corr_50vec_100m_men_anno1 = glove_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
glove_corr_50vec_150m_men_anno1 = glove_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
glove_corr_100vec_50m_men_anno1 = glove_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
glove_corr_100vec_100m_men_anno1 = glove_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
glove_corr_100vec_150m_men_anno1 = glove_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
glove_corr_150vec_50m_men_anno1 = glove_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
glove_corr_150vec_100m_men_anno1 = glove_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))
glove_corr_150vec_150m_men_anno1 = glove_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno1.tsv"))


fastText_corr_50vec_50m_men_anno2 = fastText_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
fastText_corr_50vec_100m_men_anno2 = fastText_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
fastText_corr_50vec_150m_men_anno2 = fastText_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
fastText_corr_100vec_50m_men_anno2 = fastText_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
fastText_corr_100vec_100m_men_anno2 = fastText_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
fastText_corr_100vec_150m_men_anno2 = fastText_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
fastText_corr_150vec_50m_men_anno2 = fastText_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
fastText_corr_150vec_100m_men_anno2 = fastText_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
fastText_corr_150vec_150m_men_anno2 = fastText_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))

glove_corr_50vec_50m_men_anno2 = glove_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
glove_corr_50vec_100m_men_anno2 = glove_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
glove_corr_50vec_150m_men_anno2 = glove_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
glove_corr_100vec_50m_men_anno2 = glove_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
glove_corr_100vec_100m_men_anno2 = glove_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
glove_corr_100vec_150m_men_anno2 = glove_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
glove_corr_150vec_50m_men_anno2 = glove_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
glove_corr_150vec_100m_men_anno2 = glove_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))
glove_corr_150vec_150m_men_anno2 = glove_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/men_anno2.tsv"))


fastText_corr_50vec_50m_rg = fastText_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
fastText_corr_50vec_100m_rg = fastText_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
fastText_corr_50vec_150m_rg = fastText_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
fastText_corr_100vec_50m_rg = fastText_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
fastText_corr_100vec_100m_rg = fastText_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
fastText_corr_100vec_150m_rg = fastText_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
fastText_corr_150vec_50m_rg = fastText_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
fastText_corr_150vec_100m_rg = fastText_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
fastText_corr_150vec_150m_rg = fastText_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))

glove_corr_50vec_50m_rg = glove_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
glove_corr_50vec_100m_rg = glove_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
glove_corr_50vec_150m_rg = glove_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
glove_corr_100vec_50m_rg = glove_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
glove_corr_100vec_100m_rg = glove_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
glove_corr_100vec_150m_rg = glove_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
glove_corr_150vec_50m_rg = glove_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
glove_corr_150vec_100m_rg = glove_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))
glove_corr_150vec_150m_rg = glove_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rg.tsv"))


fastText_corr_50vec_50m_rw = fastText_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
fastText_corr_50vec_100m_rw = fastText_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
fastText_corr_50vec_150m_rw = fastText_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
fastText_corr_100vec_50m_rw = fastText_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
fastText_corr_100vec_100m_rw = fastText_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
fastText_corr_100vec_150m_rw = fastText_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
fastText_corr_150vec_50m_rw = fastText_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
fastText_corr_150vec_100m_rw = fastText_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
fastText_corr_150vec_150m_rw = fastText_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))

glove_corr_50vec_50m_rw = glove_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
glove_corr_50vec_100m_rw = glove_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
glove_corr_50vec_150m_rw = glove_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
glove_corr_100vec_50m_rw = glove_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
glove_corr_100vec_100m_rw = glove_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
glove_corr_100vec_150m_rw = glove_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
glove_corr_150vec_50m_rw = glove_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
glove_corr_150vec_100m_rw = glove_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))
glove_corr_150vec_150m_rw = glove_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/rw.tsv"))


fastText_corr_50vec_50m_card = fastText_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
fastText_corr_50vec_100m_card = fastText_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
fastText_corr_50vec_150m_card = fastText_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
fastText_corr_100vec_50m_card = fastText_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
fastText_corr_100vec_100m_card = fastText_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
fastText_corr_100vec_150m_card = fastText_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
fastText_corr_150vec_50m_card = fastText_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
fastText_corr_150vec_100m_card = fastText_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
fastText_corr_150vec_150m_card = fastText_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))

glove_corr_50vec_50m_card = glove_model_50vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
glove_corr_50vec_100m_card = glove_model_50vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
glove_corr_50vec_150m_card = glove_model_50vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
glove_corr_100vec_50m_card = glove_model_100vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
glove_corr_100vec_100m_card = glove_model_100vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
glove_corr_100vec_150m_card = glove_model_100vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
glove_corr_150vec_50m_card = glove_model_150vec_50m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
glove_corr_150vec_100m_card = glove_model_150vec_100m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))
glove_corr_150vec_150m_card = glove_model_150vec_150m.evaluate_word_pairs(datapath(
    "/Users/anon_user/Developer/Coding/Projects/ml/ee-ib/code/results/sem-sim-sets/using/card.tsv"))

print("simlex fastText")

print(fastText_corr_50vec_50m_simlex)
print(fastText_corr_50vec_100m_simlex)
print(fastText_corr_50vec_150m_simlex)
print(fastText_corr_100vec_50m_simlex)
print(fastText_corr_100vec_100m_simlex)
print(fastText_corr_100vec_150m_simlex)
print(fastText_corr_150vec_50m_simlex)
print(fastText_corr_150vec_100m_simlex)
print(fastText_corr_150vec_150m_simlex)

print("simlex glove")

print(glove_corr_50vec_50m_simlex)
print(glove_corr_50vec_100m_simlex)
print(glove_corr_50vec_150m_simlex)
print(glove_corr_100vec_50m_simlex)
print(glove_corr_100vec_100m_simlex)
print(glove_corr_100vec_150m_simlex)
print(glove_corr_150vec_50m_simlex)
print(glove_corr_150vec_100m_simlex)
print(glove_corr_150vec_150m_simlex)

print("wordsim fastText")

print(fastText_corr_50vec_50m_wordsim)
print(fastText_corr_50vec_100m_wordsim)
print(fastText_corr_50vec_150m_wordsim)
print(fastText_corr_100vec_50m_wordsim)
print(fastText_corr_100vec_100m_wordsim)
print(fastText_corr_100vec_150m_wordsim)
print(fastText_corr_150vec_50m_wordsim)
print(fastText_corr_150vec_100m_wordsim)
print(fastText_corr_150vec_150m_wordsim)

print("wordsim glove")

print(glove_corr_50vec_50m_wordsim)
print(glove_corr_50vec_100m_wordsim)
print(glove_corr_50vec_150m_wordsim)
print(glove_corr_100vec_50m_wordsim)
print(glove_corr_100vec_100m_wordsim)
print(glove_corr_100vec_150m_wordsim)
print(glove_corr_150vec_50m_wordsim)
print(glove_corr_150vec_100m_wordsim)
print(glove_corr_150vec_150m_wordsim)


print("men fastText ")

print(fastText_corr_50vec_50m_men)
print(fastText_corr_50vec_100m_men)
print(fastText_corr_50vec_150m_men)
print(fastText_corr_100vec_50m_men)
print(fastText_corr_100vec_100m_men)
print(fastText_corr_100vec_150m_men)
print(fastText_corr_150vec_50m_men)
print(fastText_corr_150vec_100m_men)
print(fastText_corr_150vec_150m_men)

print("men glove")

print(glove_corr_50vec_50m_men)
print(glove_corr_50vec_100m_men)
print(glove_corr_50vec_150m_men)
print(glove_corr_100vec_50m_men)
print(glove_corr_100vec_100m_men)
print(glove_corr_100vec_150m_men)
print(glove_corr_150vec_50m_men)
print(glove_corr_150vec_100m_men)
print(glove_corr_150vec_150m_men)

print("men_anno1 fastText")

print(fastText_corr_50vec_50m_men_anno1)
print(fastText_corr_50vec_100m_men_anno1)
print(fastText_corr_50vec_150m_men_anno1)
print(fastText_corr_100vec_50m_men_anno1)
print(fastText_corr_100vec_100m_men_anno1)
print(fastText_corr_100vec_150m_men_anno1)
print(fastText_corr_150vec_50m_men_anno1)
print(fastText_corr_150vec_100m_men_anno1)
print(fastText_corr_150vec_150m_men_anno1)

print("men_anno1 glove")

print(glove_corr_50vec_50m_men_anno1)
print(glove_corr_50vec_100m_men_anno1)
print(glove_corr_50vec_150m_men_anno1)
print(glove_corr_100vec_50m_men_anno1)
print(glove_corr_100vec_100m_men_anno1)
print(glove_corr_100vec_150m_men_anno1)
print(glove_corr_150vec_50m_men_anno1)
print(glove_corr_150vec_100m_men_anno1)
print(glove_corr_150vec_150m_men_anno1)

print("men_anno2 fastText")

print(fastText_corr_50vec_50m_men_anno2)
print(fastText_corr_50vec_100m_men_anno2)
print(fastText_corr_50vec_150m_men_anno2)
print(fastText_corr_100vec_50m_men_anno2)
print(fastText_corr_100vec_100m_men_anno2)
print(fastText_corr_100vec_150m_men_anno2)
print(fastText_corr_150vec_50m_men_anno2)
print(fastText_corr_150vec_100m_men_anno2)
print(fastText_corr_150vec_150m_men_anno2)

print("men_anno1 glove")

print(glove_corr_50vec_50m_men_anno2)
print(glove_corr_50vec_100m_men_anno2)
print(glove_corr_50vec_150m_men_anno2)
print(glove_corr_100vec_50m_men_anno2)
print(glove_corr_100vec_100m_men_anno2)
print(glove_corr_100vec_150m_men_anno2)
print(glove_corr_150vec_50m_men_anno2)
print(glove_corr_150vec_100m_men_anno2)
print(glove_corr_150vec_150m_men_anno2)

print("rg fastText")

print(fastText_corr_50vec_50m_rg)
print(fastText_corr_50vec_100m_rg)
print(fastText_corr_50vec_150m_rg)
print(fastText_corr_100vec_50m_rg)
print(fastText_corr_100vec_100m_rg)
print(fastText_corr_100vec_150m_rg)
print(fastText_corr_150vec_50m_rg)
print(fastText_corr_150vec_100m_rg)
print(fastText_corr_150vec_150m_rg)

print("rg glove")

print(glove_corr_50vec_50m_rg)
print(glove_corr_50vec_100m_rg)
print(glove_corr_50vec_150m_rg)
print(glove_corr_100vec_50m_rg)
print(glove_corr_100vec_100m_rg)
print(glove_corr_100vec_150m_rg)
print(glove_corr_150vec_50m_rg)
print(glove_corr_150vec_100m_rg)
print(glove_corr_150vec_150m_rg)

print("rw fastText")

print(fastText_corr_50vec_50m_rw)
print(fastText_corr_50vec_100m_rw)
print(fastText_corr_50vec_150m_rw)
print(fastText_corr_100vec_50m_rw)
print(fastText_corr_100vec_100m_rw)
print(fastText_corr_100vec_150m_rw)
print(fastText_corr_150vec_50m_rw)
print(fastText_corr_150vec_100m_rw)
print(fastText_corr_150vec_150m_rw)

print("rw glove")

print(glove_corr_50vec_50m_rw)
print(glove_corr_50vec_100m_rw)
print(glove_corr_50vec_150m_rw)
print(glove_corr_100vec_50m_rw)
print(glove_corr_100vec_100m_rw)
print(glove_corr_100vec_150m_rw)
print(glove_corr_150vec_50m_rw)
print(glove_corr_150vec_100m_rw)
print(glove_corr_150vec_150m_rw)

print("card fastText")

print(fastText_corr_50vec_50m_card)
print(fastText_corr_50vec_100m_card)
print(fastText_corr_50vec_150m_card)
print(fastText_corr_100vec_50m_card)
print(fastText_corr_100vec_100m_card)
print(fastText_corr_100vec_150m_card)
print(fastText_corr_150vec_50m_card)
print(fastText_corr_150vec_100m_card)
print(fastText_corr_150vec_150m_card)

print("card glove")

print(glove_corr_50vec_50m_card)
print(glove_corr_50vec_100m_card)
print(glove_corr_50vec_150m_card)
print(glove_corr_100vec_50m_card)
print(glove_corr_100vec_100m_card)
print(glove_corr_100vec_150m_card)
print(glove_corr_150vec_50m_card)
print(glove_corr_150vec_100m_card)
print(glove_corr_150vec_150m_card)
