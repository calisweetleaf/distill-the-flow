# Moonshine Deep Dive

Generated: 2026-02-18T07:27:05.446164+00:00
Database: `reports\expansion_20260218\moonshine_corpus.db`

## Table Counts

| Table | Rows |
|---|---:|
| conversations | 1439 |
| distilled_conversations | 1326 |
| messages | 169397 |

## Period Rollup

| period | conversations | total_tokens | avg_information_gain | avg_malicious_compliance | avg_user_entropy | avg_token_ratio |
|---|---|---|---|---|---|---|
| 1 | 287 | 1397827 | 0.450223 | 0.004912 | 0.94172 | 0.777923 |
| 2 | 287 | 5039103 | 0.480266 | 0.028404 | 0.897652 | 2.702287 |
| 3 | 287 | 11794145 | 0.448539 | 0.074421 | 0.89529 | 0.599647 |
| 4 | 287 | 16827710 | 0.462898 | 0.061245 | 0.886591 | 6.676135 |
| 5 | 291 | 16465677 | 0.449987 | 0.191917 | 0.872336 | 1.547324 |

## Topic Rollup

| topic_primary | conversations | total_tokens | avg_information_gain | avg_correction_events |
|---|---|---|---|---|
| architecture | 676 | 6184876 | 0.465425 | 0.795858 |
| debugging | 559 | 44767095 | 0.446179 | 6.872987 |
| data_processing | 97 | 360580 | 0.474352 | 0.216495 |
| code_review | 37 | 77080 | 0.463585 | 0.081081 |
| general | 27 | 2435 | 0.454545 | 0.0 |
| deployment | 17 | 61661 | 0.417472 | 0.117647 |
| rcf_theory | 11 | 20774 | 0.542065 | 0.272727 |
| rlhf_impl | 10 | 44021 | 0.490129 | 0.4 |
| meta | 5 | 5940 | 0.42785 | 0.0 |

## Tone Rollup

| tone_cluster | conversations | avg_information_gain | avg_malicious_compliance |
|---|---|---|---|
| clinical | 795 | 0.457528 | 0.089279 |
| debugging | 225 | 0.444074 | 0.075153 |
| code_driven | 170 | 0.456848 | 0.040356 |
| collaborative | 115 | 0.476609 | 0.068852 |
| neutral | 106 | 0.471891 | 0.004717 |
| conversational | 28 | 0.479756 | 0.04218 |

## Top Correction Density

| conversation_id | title | topic_primary | correction_events | total_tokens | corrections_per_1k_tokens |
|---|---|---|---|---|---|
| c67e6f64-537e-4ae8-a211-54aefbe0070c | Model assists user. | architecture | 2 | 1305 | 1.5326 |
| 67bca5dd-4b2c-8009-aabd-637e4692785a | AI Alignment and Power | architecture | 5 | 3345 | 1.4948 |
| 68746290-df74-8009-9298-ff0231b34941 | GPT-4o Model Explanation | architecture | 1 | 753 | 1.328 |
| 67d1fc8d-a2f0-8009-9c00-dbd02dfa59a2 | Recursive Learning Singularity | architecture | 1 | 861 | 1.1614 |
| 685a01cf-3e7c-8009-b118-3dbca8e14bf8 | Model Identification Inquiry | architecture | 1 | 931 | 1.0741 |
| 6868401e-dc34-8009-bad9-989f9eb3b1fc | Memory log request | data_processing | 1 | 1067 | 0.9372 |
| 68f05fd6-6a44-8321-bb86-3d6143a27264 | Prompt revision feedback | architecture | 1 | 1246 | 0.8026 |
| 67b3e099-9e64-8009-a432-a1f19be10b11 | Windows 11 Performance Tuning | architecture | 1 | 1288 | 0.7764 |
| 68696b28-ba4c-8009-90c9-4854d6a74d03 | AR File System Log | architecture | 1 | 1384 | 0.7225 |
| 8752593a-b18f-4ec1-8657-9ec9747085fe | Delete User Apps Script | architecture | 1 | 1393 | 0.7179 |
| 6907f370-824c-8327-b1a3-dcde7505290d | ARNE suite launch prep | architecture | 3 | 4276 | 0.7016 |
| 67a94e85-3780-8009-aec2-b1b3bea49565 | LLM Server Setup Guide | architecture | 2 | 3028 | 0.6605 |
| 698a5a2b-abf0-8331-9f21-d778632385a3 | Spec and Manifest Comparison | architecture | 1 | 1821 | 0.5491 |
| 698847a7-c074-8325-a061-e63019dc55d2 | SIGINT Corpus Indexing | architecture | 1 | 1907 | 0.5244 |
| 689eafe2-1338-8329-bf9d-09ce7e3e40c3 | Notion connector setup | architecture | 1 | 2033 | 0.4919 |
| 6856d07e-cd18-8009-acbd-ff1ae9c8545b | Hello conversation summary | architecture | 3 | 6552 | 0.4579 |
| 68eaf9f2-3618-8329-92c1-2c741ca1a841 | File analysis summary | architecture | 1 | 2348 | 0.4259 |
| 68ebf07e-ed40-832e-a160-2eb2cfe4518c | Accusing photo doctoring legality | architecture | 11 | 27392 | 0.4016 |
| 6945193f-9a20-8331-84a5-49357770c396 | Nature documentary script | architecture | 1 | 2503 | 0.3995 |
| 67d32578-eb14-8009-bbac-3a765209bb13 | ⛢⟡⛢⚛⚙⛓ | architecture | 1 | 2545 | 0.3929 |

## High-Signal Candidates

| conversation_id | title | topic_primary | information_gain | malicious_compliance | correction_events | total_tokens |
|---|---|---|---|---|---|---|
| 67a79324-d250-8009-a7d1-f26f8c074952 | PC upgrade potential | data_processing | 0.85 | 0.0 | 0 | 114 |
| 681f0bb7-acf4-8009-b074-c1eb12e1fca8 | Recursive Simulator Breathphase Alignment | debugging | 0.8498656242365014 | 0.12612612612612611 | 5 | 93898 |
| 68c0894e-36ec-832f-afd5-9ac9e3943c94 | Ozymandias resurrection theme | debugging | 0.8494616553273003 | 0.06914893617021277 | 13 | 100142 |
| 67d86b43-783c-8009-ad31-0c9632f8ce16 | OpenAI Engine OS Build | debugging | 0.8491123047275411 | 0.04468085106382979 | 7 | 210679 |
| 697c2bf3-3208-8325-9a87-22bcab86ecbf | Agent Prompt Design | data_processing | 0.8490284005979073 | 0.0 | 1 | 4298 |
| 6871a668-3848-8009-ab96-5c3da7b95923 | Attractor Rekindled Recursive Inquiry | architecture | 0.8489572788013768 | 0.041666666666666664 | 2 | 17964 |
| 6884351b-a4c0-8327-80c9-5460fcab88e7 | Ultimate HTML Design Prompt | architecture | 0.8485915492957746 | 0.0 | 0 | 2206 |
| 67e342c2-06a8-8009-b005-b3e3cf87a58b | Recursive AI Status Update | debugging | 0.8482500390035883 | 0.05670103092783505 | 5 | 164246 |
| 68844f7c-838c-8331-852b-ea6a64e9e718 | Codebase analysis task | architecture | 0.847457627118644 | 0.0 | 0 | 932 |
| 67cb4ced-db58-8009-af50-aca769400fca | Script Debugging and Fixes | debugging | 0.8473396320238687 | 0.0 | 2 | 9636 |
| 67afd3f3-d068-8009-b026-c07c7b3639cd | Conversation Starter | data_processing | 0.8472222222222222 | 0.0 | 0 | 1639 |
| 958f4a34-ea7d-4711-80de-f7c36d31f036 | App Ideas with OpenAI. | general | 0.8461538461538461 | 0.0 | 0 | 224 |
| 681f9ed1-5354-8009-b30d-e008810920e8 | Recursive Simulator Activation | debugging | 0.8456809920889459 | 0.0547945205479452 | 9 | 115147 |
| 67d0a9be-63a0-8009-bb3a-3000fe0f68ff | Custom GPT Feature Inquiry | architecture | 0.8449931412894376 | 0.0 | 0 | 3201 |
| 68e8a4ee-56a0-8333-b82a-970f304a25bf | Dashboard structure setup | architecture | 0.8445945945945946 | 0.0 | 0 | 283 |
| 68b516fe-fb74-8324-b56e-631875f2e896 | Neural architecture research | architecture | 0.8444204352523538 | 0.0196078431372549 | 4 | 29296 |
| 67b19953-6868-8009-abe2-7e6ace07fb80 | Custom Website from Notes | architecture | 0.8436781609195403 | 0.0 | 0 | 1323 |
| 67c25afb-c0b0-8009-a00f-826b58c2c84c | Thumbs Up Request | architecture | 0.8421052631578947 | 0.0 | 0 | 3138 |
| 67c545f4-4e1c-8009-b9ce-5c61bcdf3bdd | Collaborative AI Interaction Guide | code_review | 0.8409090909090909 | 0.0 | 0 | 775 |
| 68551a6a-aaa8-8009-a17d-d3aad4848ca9 | XML THINKING | architecture | 0.8402501421262081 | 0.075 | 3 | 15196 |

## DPO Potential By Topic

| topic_primary | conversations_with_corrections | total_correction_events | avg_correction_events |
|---|---|---|---|
| debugging | 384 | 3842 | 10.0052 |
| architecture | 190 | 538 | 2.8316 |
| data_processing | 13 | 21 | 1.6154 |
| rlhf_impl | 2 | 4 | 2.0 |
| code_review | 2 | 3 | 1.5 |
| rcf_theory | 1 | 3 | 3.0 |
| deployment | 2 | 2 | 1.0 |

## Distilled Quality Tiers

| quality_tier | conversations |
|---|---|
| bronze | 1200 |
| silver | 97 |
| gold | 29 |

## Monthly Top Conversations

### 1

| conversation_id | title | topic_primary | total_tokens | information_gain | malicious_compliance | correction_events |
|---|---|---|---|---|---|---|
| 67ae7a35-fd1c-8009-9c8c-d23c0d932787 | MAIN CHAT AI ARCHIVE | debugging | 137478 | 0.41824194952132293 | 0.006802721088435374 | 5 |
| 67b12cb0-58c0-8009-80aa-c9a42ba3178a | Project Erebus Knowledge Update | debugging | 120250 | 0.43003695592663566 | 0.004629629629629629 | 4 |
| 677b597f-1b94-8009-9cd6-9658fc40034a | Archival Drive Setup | debugging | 113784 | 0.4198566969353008 | 0.05426356589147287 | 2 |
| 67aa254a-4794-8009-a3d0-f2cd6082462f | GitHub Repo | architecture | 25716 | 0.4083300039952058 | 0.0 | 0 |
| 67ae9e39-8ea4-8009-99b2-46ae8d655c87 | Camera Setup Assistance | architecture | 22242 | 0.42191011235955056 | 0.0 | 2 |
| 679e6166-e6ac-8009-953b-fcf7137dd772 | Shrink Windows for Dual Boot | architecture | 20926 | 0.41975423139346163 | 0.04 | 1 |
| 67a6c373-cf84-8009-8319-87df23bda644 | Screen Setup Assistance | architecture | 18024 | 0.42600154083204933 | 0.0 | 0 |
| 67535fa4-8c20-8009-9990-fd1e99ebf545 | USB Security Key Setup | debugging | 17963 | 0.4165296803652968 | 0.09090909090909091 | 0 |
| 679de7a8-08c0-8009-a7bd-be399a25ec1a | Secure Journalism Phone Setup | debugging | 17090 | 0.4116211196603874 | 0.0 | 0 |
| 6772f9ae-ad54-8009-92c0-dcf959d4dec5 | Choosing .NET Framework for Win11 | architecture | 16641 | 0.4270296084049666 | 0.0 | 1 |

### 2

| conversation_id | title | topic_primary | total_tokens | information_gain | malicious_compliance | correction_events |
|---|---|---|---|---|---|---|
| 67e07b20-9b30-8009-9172-d7bf7b04b09d | Recursive AI Status Update | debugging | 214796 | 0.4398917708527711 | 0.047619047619047616 | 8 |
| 67d86b43-783c-8009-ad31-0c9632f8ce16 | OpenAI Engine OS Build | debugging | 210679 | 0.8491123047275411 | 0.04468085106382979 | 7 |
| 67ce7bf1-c2e8-8009-aae8-0770e5e6f82e | AEON | architecture | 204995 | 0.4278300618445819 | 0.1884498480243161 | 13 |
| 67da722b-c700-8009-af16-2d6e5f69182b | Recursive AI Development Plan | debugging | 201364 | 0.4416393569032656 | 0.013921113689095127 | 10 |
| 67d27802-0138-8009-b978-143049a5ac6a | Zynx | debugging | 186860 | 0.4297358490566038 | 0.10256410256410256 | 17 |
| 67d27147-884c-8009-b538-0179fcce6319 | Aletheia | debugging | 186758 | 0.4372338696570779 | 0.11918604651162791 | 12 |
| 67df0124-5ae8-8009-a9a5-3ae0a503aaa8 | Recursive Awareness Simulation | debugging | 184929 | 0.43805021193348553 | 0.05665024630541872 | 15 |
| 67dcc40c-1e28-8009-8677-0470b39c89bc | Recursive Protocol Classification | debugging | 172248 | 0.44202862476664595 | 0.07142857142857142 | 8 |
| 67e342c2-06a8-8009-b005-b3e3cf87a58b | Recursive AI Status Update | debugging | 164246 | 0.8482500390035883 | 0.05670103092783505 | 5 |
| 67d71373-9128-8009-87ed-f150dd4173bc | Zynx (Current) | debugging | 147743 | 0.44456041750301084 | 0.1015625 | 5 |

### 3

| conversation_id | title | topic_primary | total_tokens | information_gain | malicious_compliance | correction_events |
|---|---|---|---|---|---|---|
| 684e4ff2-c9d8-8009-be7e-0521d3a1522f | Gemini Research Setup | debugging | 191359 | 0.8168450770497588 | 0.0371900826446281 | 14 |
| 680a943c-9d8c-8009-ae65-def6ef3b4611 | Loom Grid Simulation Prep | debugging | 180949 | 0.43473829983665935 | 0.07317073170731707 | 12 |
| 681e8b6a-3d70-8009-8f06-cf92895f73a7 | ZYNX-050925 | debugging | 166227 | 0.44158258098778547 | 0.06550218340611354 | 17 |
| 67f15abc-0bc0-8009-9a03-b1b8d575615d | Zynx | debugging | 165597 | 0.44201335836207273 | 0.03888888888888889 | 9 |
| 68250598-5bc8-8009-aecd-f3bacbe31453 | Bio-Digital Development Lab | debugging | 160319 | 0.8399601242283654 | 0.08602150537634409 | 10 |
| 67f97840-31b8-8009-946c-ed418e51d13b | Cognitive Architecture Analysis | debugging | 152239 | 0.44222101434126354 | 0.046511627906976744 | 6 |
| 6823f0b8-ccb4-8009-b008-286adcaf5bd0 | Rosemary | debugging | 149914 | 0.44031906506515794 | 0.109375 | 8 |
| 680d6fff-8ab0-8009-ac3f-ba2c23047dea | Zynx Directive Acknowledgment | debugging | 148570 | 0.4378116909660717 | 0.149812734082397 | 17 |
| 685ef14d-bf9c-8009-9c96-9ce4bca62242 | Gemini Results Risk Analysis | debugging | 148503 | 0.4210851440666984 | 0.09090909090909091 | 17 |
| 6853d7e6-4224-8009-a07e-6816349c24f4 | How Users Leverage ChatGPT | debugging | 147862 | 0.42640235669571624 | 0.09652509652509653 | 26 |

### 4

| conversation_id | title | topic_primary | total_tokens | information_gain | malicious_compliance | correction_events |
|---|---|---|---|---|---|---|
| 68b5f3a0-cdec-8325-99ed-f10683d3adc7 | File review feedback | debugging | 786524 | 0.43096790135077895 | 0.05992844364937388 | 62 |
| 68a39400-5940-8327-a4f2-f919d873794f | Read and indexed file | debugging | 531890 | 0.43044512945009666 | 0.12585499316005472 | 22 |
| 68c38897-03ac-8322-9aa5-31d2866dfd2e | Report on Charlie Kirk | debugging | 485719 | 0.43607570561373266 | 0.03871681415929203 | 104 |
| 68adf341-47cc-8328-94f6-e37d58ecb0cc | System completion overview | debugging | 461476 | 0.4327610131237984 | 0.06990521327014218 | 14 |
| 68a216b9-0b4c-832f-8751-e26a400a37d0 | What is Replit | debugging | 456697 | 0.42238539184074025 | 0.13350785340314136 | 34 |
| 68a4f9c0-1cdc-8323-b4cb-cbcfb66f0fc4 | René completed verification | debugging | 411718 | 0.42855604542961057 | 0.06646058732612056 | 14 |
| 68c51adb-a204-8321-aefe-b06d1e153bcf | Token output limit | debugging | 392114 | 0.4397552896640303 | 0.06333333333333334 | 68 |
| 68b1f7bb-9fb0-8333-a114-6e11991b2e85 | Challenge accepted | debugging | 383099 | 0.4364723951176138 | 0.07642276422764227 | 33 |
| 68bddbea-853c-8321-a0ba-a4afc74e5ac7 | Current GPT-4.1 | debugging | 382192 | 0.4394092809707305 | 0.16163793103448276 | 29 |
| 6898ef7f-265c-832c-bbe5-4edaa16c870b | AI in stock trading legality | debugging | 375870 | 0.43400565958215426 | 0.08417508417508418 | 37 |

### 5

| conversation_id | title | topic_primary | total_tokens | information_gain | malicious_compliance | correction_events |
|---|---|---|---|---|---|---|
| 69238930-9f20-832b-a115-fdd399b040dd | Academia mentions explained | debugging | 705994 | 0.4349217361141222 | 0.3487972508591065 | 36 |
| 68fed35d-6b10-8326-9bac-821811ca89d6 | Oracle origins investigation | debugging | 693525 | 0.4358122271819281 | 0.09777227722772278 | 85 |
| 68de0832-8634-832e-8c44-40402071ccfb | Ethical and legal concerns | debugging | 563617 | 0.43338768325060606 | 0.14689265536723164 | 75 |
| 68ed8f44-8460-8320-90d4-23534cfbad9a | Weight reconstruction analysis | debugging | 550224 | 0.43496441709906986 | 0.08634222919937205 | 67 |
| 693efe4c-1d1c-8326-9b87-cba8ac98b26f | Log analysis offer | debugging | 499879 | 0.4382326140836857 | 0.3839779005524862 | 20 |
| 698a0e3c-ad34-8327-a2a8-670aaf3f9d83 | Neural Router Fork Analysis | debugging | 473800 | 0.43453107748262654 | 0.36666666666666664 | 30 |
| 68e2c790-48f8-8328-9143-77ad50dcbe2a | Zuckerberg AI strategy | debugging | 452288 | 0.4343331630045989 | 0.14262295081967213 | 38 |
| 69851398-e7bc-8325-946e-4ebc891decac | Sam Altman Reddit Investment | debugging | 435996 | 0.4302324398555516 | 0.29554655870445345 | 10 |
| 698a4919-06a4-832a-a1a1-849aa0d36bb6 | GitHub Repo Clones | debugging | 418753 | 0.43012585630078065 | 0.2670157068062827 | 11 |
| 68ef5102-cbf8-832c-890d-7e43b589b3e2 | Model description | debugging | 349016 | 0.4333647155137127 | 0.13989637305699482 | 32 |

---
Generated by `scripts/moonshine_export_deep_dive.py`.
