# GAT-GO:Accurate Protein Function Prediction via Graph Attention Networks with Predicted Structure Information
[[paper]](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbab502/6457163)
 # Citation
This is the official code repository for the paper "Accurate Protein Function Prediction via Graph Attention Networks with Predicted Structure Information".
# Dependencies
```pytorch >=1.7.1```  
```pytorch-geometric >= 1.7.0```  

# Predict sequence function with GAT-GO
To use GAT-GO with pre-processed data, we provided a ```data_loader``` which parses the pre-processed sequence features ussed in GAT-GO.  
```GAT-GO.py``` can be used to make prediction with pre-processed seuqnces. Examples can be found in this [Notebook](link)  
```
python GAT-GO.py --ModelPath <PATH> --Device <CUDA device> --BatchSize <BatchSize> --SeqIDs <SeqIDs> --OutDir <OutDir>
```  
\*\*\*\* Intput \*\*\*\* 
```
--ModelPath : Path to trained model weights
--OutDir    : Output directory, where the result will be saved
--BatchSize : Batch size  
--Device    : CUDA device to be used for inferece
```
\*\*\*\* Output \*\*\*\*  
```GAT-GO_Results.pt``` will be saved at ```<OutDir>``` which is a serialized dictionary indexed by sequence identifiers provided in ```<SeqIDs>```

To extract the GO-terms from the result, please see the [Notebook](link) example. 
# Data Format
For each sequence in PDB/PDBmmseq dataset, a serialized dictionary stores the processed features used in GAT-GO. Details can be found below  
1. ```data.seq: One-hot encoded primary sequence```  
2. ```data.pssm: Sequence profile constructed from MSA```
3. ```data.x: Residue level sequence embedding generated from ESM-1b```
4. ```data.edge_index: Contact map index```
5. ```data.seq_embed: Sequence level embedding generated from ESMA-1b```
6. ```data.label: GO term annotation```
7. ```data.chain_id: Sequence identifier```

# Data & Pre-trained model
