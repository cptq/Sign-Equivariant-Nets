echo "Erdos-Renyi graphs"
python main_link_pred.py --model gcn --use_eigs 0 --num_eigs 0 --graph_type er
python main_link_pred.py --model signnet --use_eigs 1 --graph_type er
python main_link_pred.py --model decode_only --use_eigs 1 --graph_type er
python main_link_pred.py --model learn_decode --use_eigs 1 --graph_type er
python main_link_pred.py --model sign_equiv --use_eigs 1 --graph_type er

echo "Barabasi-Albert graphs"
python main_link_pred.py --model gcn --use_eigs 0 --num_eigs 0 --graph_type pa
python main_link_pred.py --model signnet --use_eigs 1 --graph_type pa
python main_link_pred.py --model decode_only --use_eigs 1 --graph_type pa
python main_link_pred.py --model learn_decode --use_eigs 1 --graph_type pa
python main_link_pred.py --model sign_equiv --use_eigs 1 --graph_type pa