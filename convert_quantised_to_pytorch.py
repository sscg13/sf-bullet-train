import struct
import argparse

FULL_THREATS_HASH = 0x8F234CB8
HALFKA_V2_HM_HASH = 0x7F234CB8
FEATURE_SET_HASH = (((FULL_THREATS_HASH << 1) | (FULL_THREATS_HASH >> 31)) & 0xFFFFFFFF) ^ HALFKA_V2_HM_HASH
VERSION = 0x6A448AFA
DEFAULT_DESCRIPTION = "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer."
HALFKA_V2_REAL_FEATURES = 704 * 32
FULL_THREATS_FEATURES = 60_720
FEATURE_TRANSFORMER_FEATURES = HALFKA_V2_REAL_FEATURES + FULL_THREATS_FEATURES

def encode_leb_128_array(arr):
    res = []
    for v in arr:
        while True:
            byte = v & 0x7f
            v = v >> 7
            if (v == 0 and byte & 0x40 == 0) or (v == -1 and byte & 0x40 != 0):
                res.append(byte)
                break
            res.append(byte | 0x80)
    return res

def fc_hash(L1, L2, L3, num_ls_buckets=8):
    # InputSlice hash
    prev_hash = 0xEC42E90D
    prev_hash ^= (L1 * 2)

    # Fully connected layers
    layers_info = [
        (L2 + 1, True), 
        (L3, True),      
        (1, False)      
    ]
    
    for out_features, has_relu in layers_info:
        layer_hash = 0xCC03DAE4
        layer_hash += out_features // num_ls_buckets
        layer_hash ^= prev_hash >> 1
        layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
        if out_features // num_ls_buckets != 1 and has_relu:
            # Clipped ReLU hash
            layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
        prev_hash = layer_hash
    
    return prev_hash

def write_header(outfile, L1, fc_hash_val, description=DEFAULT_DESCRIPTION):
    # Write version
    outfile.write(struct.pack('<I', VERSION))
    
    # Write network hash
    network_hash = fc_hash_val ^ FEATURE_SET_HASH ^ (L1 * 2)
    outfile.write(struct.pack('<I', network_hash))
    
    # Write description
    encoded_description = description.encode('utf-8')
    outfile.write(struct.pack('<I', len(encoded_description)))
    outfile.write(encoded_description)

def write_leb_128_array(outfile, arr):
    buf = encode_leb_128_array(arr)
    outfile.write(struct.pack('<I', len(buf)))
    outfile.write(bytes(buf))

def read_binary_file(filename, L1, L2, L3, num_buckets=8):
    with open(filename, 'rb') as f:
        data = {}
        
        # Read l0b (16-bit) - L1 values
        data['l0b'] = list(struct.unpack('<' + 'h' * L1, f.read(L1 * 2)))
        
        # Read l0p (16-bit) - HalfKAv2_hm export weights
        data['l0p'] = list(struct.unpack('<' + 'h' * (HALFKA_V2_REAL_FEATURES * L1), f.read(HALFKA_V2_REAL_FEATURES * L1 * 2)))
        
        # Read l0t (16-bit) - Full_Threats weights
        data['l0t'] = list(struct.unpack('<' + 'h' * (FULL_THREATS_FEATURES * L1), f.read(FULL_THREATS_FEATURES * L1 * 2)))
        
        # Read pst (32-bit) - feature PSQT weights
        data['pst'] = list(struct.unpack('<' + 'i' * (FEATURE_TRANSFORMER_FEATURES * num_buckets), f.read(FEATURE_TRANSFORMER_FEATURES * num_buckets * 4)))
        
        # Read l1b (32-bit) - ((L2 + 1) * num_buckets) values 
        data['l1b'] = list(struct.unpack('<' + 'i' * ((L2 + 1) * num_buckets), f.read((L2 + 1) * num_buckets * 4)))
        
        # Read l1w (8-bit) - (L1 * (L2 + 1) * num_buckets) values 
        data['l1w'] = list(struct.unpack('<' + 'b' * (L1 * (L2 + 1) * num_buckets), f.read(L1 * (L2 + 1) * num_buckets)))
        
        # Read l2b (32-bit) - (L3 * num_buckets) values 
        data['l2b'] = list(struct.unpack('<' + 'i' * (L3 * num_buckets), f.read(L3 * num_buckets * 4)))
        
        # Read l2w (8-bit) - (L2 * 2 * L3 * num_buckets) values 
        data['l2w'] = list(struct.unpack('<' + 'b' * (L2 * 2 * L3 * num_buckets), f.read(L2 * 2 * L3 * num_buckets)))
        
        # Read l3b (32-bit) - num_buckets values 
        data['l3b'] = list(struct.unpack('<' + 'i' * num_buckets, f.read(num_buckets * 4)))
        
        # Read l3w (8-bit) - (L3 * num_buckets) values 
        data['l3w'] = list(struct.unpack('<' + 'b' * (L3 * num_buckets), f.read(L3 * num_buckets)))
        
        return data

def organize_into_buckets(data, L1, L2, L3, num_buckets=8):
    bucketed_data = {
        'l0b': data['l0b'],  # No bucketing for l0b
        'l0t': [],  
        'l0p': data['l0p'],  # No bucketing for l0p
        'pst_t': [],
        'pst_p': [],
        'l1': [],
        'l2': [],
        'l3': []
    }
    # Clip threat weights
    for weight_idx in range(FULL_THREATS_FEATURES * L1):
        bucketed_data['l0t'].append(max(min(data['l0t'][weight_idx], 127), -128))
        
    # Bullet writes HalfKAv2_hm first and Full_Threats second. nnue-pytorch
    # serializes the composed feature set as Full_Threats+HalfKAv2_hm^.
    threat_start = HALFKA_V2_REAL_FEATURES * num_buckets
    threat_end = FEATURE_TRANSFORMER_FEATURES * num_buckets
    psq_start = 0
    psq_end = HALFKA_V2_REAL_FEATURES * num_buckets
    bucketed_data['pst_t'].extend(data['pst'][threat_start:threat_end])
    bucketed_data['pst_p'].extend(data['pst'][psq_start:psq_end])
    
    # Organize l1 layer (bias and weights) into buckets
    for bucket in range(num_buckets):
        bias_start = bucket * (L2 + 1)
        bias_end = bias_start + (L2 + 1)
        weights_start = bucket * L1 * (L2 + 1)
        weights_end = weights_start + L1 * (L2 + 1)
        
        bucketed_data['l1'].append({
            'bias': data['l1b'][bias_start:bias_end],
            'weights': data['l1w'][weights_start:weights_end]
        })
    
    # Organize l2 layer (bias and weights) into buckets
    for bucket in range(num_buckets):
        bias_start = bucket * L3
        bias_end = bias_start + L3
        weights_start = bucket * L2 * 2 * L3
        weights_end = weights_start + L2 * 2 * L3

        weights = data['l2w'][weights_start:weights_end]
        
        # EDIT: I think this needs to be aligned to 32 elements, so add two zeros after every L2*2 (30) elements
        # Insert two zeros after every L2*2 (30) elements
        modified_weights = []
        for i in range(0, len(weights), L2 * 2):
            chunk = weights[i:i + L2 * 2]
            modified_weights.extend(chunk)
            modified_weights.extend([0, 0])
        
        bucketed_data['l2'].append({
            'bias': data['l2b'][bias_start:bias_end],
            'weights': modified_weights
        })
    
    # Organize l3 layer (bias and weights) into buckets
    for bucket in range(num_buckets):
        bias_start = bucket
        bias_end = bias_start + 1
        weights_start = bucket * L3
        weights_end = weights_start + L3
        
        bucketed_data['l3'].append({
            'bias': data['l3b'][bias_start:bias_end],
            'weights': data['l3w'][weights_start:weights_end]
        })
    
    return bucketed_data

def convert_binary_format(input_file, output_file, L1=128, L2=15, L3=32, num_buckets=8):
    # Read the input binary file
    data = read_binary_file(input_file, L1, L2, L3, num_buckets)

    print(f"Read {input_file} successfully.")
    
    # Organize data into buckets
    bucketed_data = organize_into_buckets(data, L1, L2, L3, num_buckets)

    print(f"Organized data into {num_buckets} buckets.")
    
    # Calculate hashes
    # fc_hash_val = fc_hash(L1, L2, L3, num_buckets)

    print(f"Writing to {output_file}...")
    
    with open(output_file, 'wb') as outfile:
        # Write header
        fc_hash_val = fc_hash(L1, L2, L3, num_buckets)

        write_header(outfile, L1, fc_hash_val)
        
        # Write feature transformer hash
        feature_transformer_hash = FEATURE_SET_HASH ^ (L1 * 2)
        outfile.write(struct.pack('<I', feature_transformer_hash))
        
        # Write feature transformer in nnue-pytorch's composed-feature order:
        # bias, Full_Threats weights, Full_Threats PSQT, HalfKAv2_hm weights,
        # HalfKAv2_hm PSQT.
        outfile.write('COMPRESSED_LEB128'.encode('utf-8'))
        write_leb_128_array(outfile, bucketed_data['l0b'])
        
        outfile.write(struct.pack('<' + 'b' * len(bucketed_data['l0t']), *bucketed_data['l0t']))

        outfile.write('COMPRESSED_LEB128'.encode('utf-8'))
        write_leb_128_array(outfile, bucketed_data['pst_t'])
        
        outfile.write('COMPRESSED_LEB128'.encode('utf-8'))
        write_leb_128_array(outfile, bucketed_data['l0p'])
        
        outfile.write('COMPRESSED_LEB128'.encode('utf-8'))
        write_leb_128_array(outfile, bucketed_data['pst_p'])
        
        # Write each bucket of l1, l2 and l3 layers
        for bucket in range(num_buckets):
            outfile.write(struct.pack('<I', fc_hash_val))
            
            assert len(bucketed_data['l1'][bucket]['bias']) == (L2 + 1)
            assert len(bucketed_data['l1'][bucket]['weights']) == L1 * (L2 + 1)

            outfile.write(struct.pack('<' + 'i' * len(bucketed_data['l1'][bucket]['bias']), 
                                     *bucketed_data['l1'][bucket]['bias']))
            outfile.write(struct.pack('<' + 'b' * len(bucketed_data['l1'][bucket]['weights']), 
                                     *bucketed_data['l1'][bucket]['weights']))
            
            start_position = outfile.tell()
            
            assert len(bucketed_data['l2'][bucket]['bias']) == L3
            assert len(bucketed_data['l2'][bucket]['weights']) == (L2 + 1) * 2 * L3 
            
            outfile.write(struct.pack('<' + 'i' * len(bucketed_data['l2'][bucket]['bias']), 
                                     *bucketed_data['l2'][bucket]['bias']))
            outfile.write(struct.pack('<' + 'b' * len(bucketed_data['l2'][bucket]['weights']), 
                                     *bucketed_data['l2'][bucket]['weights']))
            
            end_position = outfile.tell()
            bucket_size = end_position - start_position

            print(f"Ending position for bucket {bucket}: {end_position}")
            print(f"Bucket {bucket} size: {bucket_size} bytes")

            assert len(bucketed_data['l3'][bucket]['bias']) == 1
            assert len(bucketed_data['l3'][bucket]['weights']) == L3

            outfile.write(struct.pack('<' + 'i' * len(bucketed_data['l3'][bucket]['bias']), 
                                     *bucketed_data['l3'][bucket]['bias']))
            outfile.write(struct.pack('<' + 'b' * len(bucketed_data['l3'][bucket]['weights']), 
                                     *bucketed_data['l3'][bucket]['weights']))



def main():
    parser = argparse.ArgumentParser(description='Convert binary neural network format')
    parser.add_argument('input_file', type=str, help='Input binary file path')
    parser.add_argument('output_file', type=str, help='Output binary file path')
    parser.add_argument('--L1', type=int, default=128, help='L1 parameter (default: 128)')
    parser.add_argument('--L2', type=int, default=15, help='L2 parameter (default: 15)')
    parser.add_argument('--L3', type=int, default=32, help='L3 parameter (default: 32)')
    parser.add_argument('--buckets', type=int, default=8, help='Number of buckets (default: 8)')
    
    args = parser.parse_args()
    
    convert_binary_format(args.input_file, args.output_file, args.L1, args.L2, args.L3, args.buckets)
    print(f"Conversion complete: {args.input_file} -> {args.output_file}")


if __name__ == "__main__":
    main()
