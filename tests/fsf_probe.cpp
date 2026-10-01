// Compatibility probe linked against the unmodified Fairy-Stockfish objects.
// Build instructions are in README.md. Compile with the engine's SIMD flags.
#include <fstream>
#include <iostream>
#include <string>
#include "bitboard.h"
#include "evaluate.h"
#include "piece.h"
#include "position.h"
#include "nnue/evaluate_nnue.h"
#include "psqt.h"
#include "variant.h"

namespace Stockfish::Eval::NNUE {
extern LargePagePtr<FeatureTransformer> featureTransformer;
extern AlignedPtr<Network> network[LayerStacks];
}

int main(int argc, char** argv) {
    using namespace Stockfish;
    namespace NN = Eval::NNUE;
    if (argc != 2) return 2;
    pieceMap.init();
    variants.init();
    auto variant = variants.find("shatranj")->second;
    currentNnueVariant = variant;
    PSQT::init(variant);
    Bitboards::init();
    Position::init();
    std::ifstream net(argv[1], std::ios::binary);
    if (!NN::load_eval(argv[1], net)) {
        std::cerr << "Fairy-Stockfish rejected the network\n";
        return 1;
    }
    std::cerr << "dims=" << variant->nnueDimensions
              << " hash=" << NN::HashValue << " fc=" << NN::Network::get_hash_value() << '\n';
    std::string fen;
    while (std::getline(std::cin, fen)) {
        StateInfo state{};
        Position position;
        position.set(variant, fen, false, &state, nullptr);
        const auto bucket = std::min((position.count<ALL_PIECES>() - 1) * 8 / variant->nnueMaxPieces, 7);
        std::cout << "{\"white\":[";
        for (Color colour : {WHITE, BLACK}) {
            ValueList<NN::IndexType, 128> active;
            NN::FeatureSet::append_active_indices(position, colour, active);
            for (std::size_t i = 0; i < active.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << active[i];
            }
            std::cout << (colour == WHITE ? "],\"black\":[" : "],");
        }
        alignas(NN::CacheLineSize) NN::TransformedFeatureType features[NN::FeatureTransformer::BufferSize];
        alignas(NN::CacheLineSize) char buffer[NN::Network::BufferSize];
        const auto psqt = NN::featureTransformer->transform(position, features, bucket);
        const auto dense = NN::network[bucket]->propagate(features, buffer)[0];
        std::cout << "\"bucket\":" << bucket << ",\"psqt\":" << psqt
                  << ",\"dense\":" << dense << ",\"value\":" << NN::evaluate(position, false) << "}\n";
    }
    variants.clear_all();
    pieceMap.clear_all();
}
