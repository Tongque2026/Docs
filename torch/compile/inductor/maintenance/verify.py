"""检查正式文档引用、模块关系和已有实验证据；不代表源码覆盖完成。"""
from pathlib import Path
import ast,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
state=json.loads((ROOT/'maintenance/state.json').read_text())
nodes={n['id']:n for n in state['nodes']}
assert len(nodes)==len(state['nodes'])
for n in nodes.values():
    children=[c for c in nodes.values() if c['parent']==n['id']]
    assert len(children)<=4,(n['id'],len(children))
    assert n['parent'] is None or n['parent'] in nodes
    if n['status'] in ('本层完成','叶子·逐行完成'):
        article=ROOT/n['path'];assert article.is_file()
        match=re.search(r'```text\n(.*?)\n```',article.read_text(),flags=re.S)
        assert match and '← 当前篇' in match.group(1),article
        assert '<!--' not in match.group(1),article
q=[];work=['inductor']
while work:
    nid=work.pop(0);n=nodes[nid]
    if n['status']=='待展开':q.append(nid)
    work.extend(c['id'] for c in nodes.values() if c['parent']==nid)
assert q==state['breadth_first_queue']
focus=state.get('priority_focus',{}).get('nodes',[])
assert len(focus)==len(set(focus)) and all(nid in nodes for nid in focus)
def focus_rank(nid):
    return next((i for i,root in enumerate(focus) if nid==root or nid.startswith(root+'/')),None)
priority=sorted((nid for nid in q if focus_rank(nid) is not None),key=lambda nid:(nid.count('/'),focus_rank(nid),q.index(nid)))
reading=priority+[nid for nid in q if nid not in priority]
assert state['reading_queue']==reading and len(reading)==len(set(reading))
assert state['next']==(reading[0] if reading else None)
pattern=json.loads((ROOT/'experiments/pattern-protocol/results.json').read_text())
assert len(pattern)==12
assert [pattern[k]['count'] for k in ['graph_success','graph_extra_false','inner_multiuser','root_multiuser','mutation_barrier']]==[1,0,0,1,0]
assert pattern['inner_multiuser']['events']==pattern['mutation_barrier']['events']==[]
assert pattern['graph_success']['events'][1]['arg_type']=='Node'
assert pattern['named_same']['matched'] and not pattern['named_different']['matched']
assert pattern['same_root_entries']['count']==2 and pattern['same_root_entries']['events']==['observe_without_edit','edit']
assert pattern['candidate_snapshot']['count']==1
assert [e['phase'] for e in pattern['replacement_entry']['application_events']]==['trace','specific_extra','trace']
assert pattern['replacement_entry']['targets']==['aten.mul.Tensor']
lower=pattern['lowering_entry']
assert (lower['calls_after_apply'],lower['calls_after_compile'],lower['calls_after_runtime'])==(0,1,1)
assert lower['events'][0]['arg_type']=='TensorBox'
assert pattern['lowered_graph_repropagation']['error']=='StopIteration'
assert pattern['lowered_graph_repropagation']['events'][0]['arg_type']=='FakeTensor'
batch=json.loads((ROOT/'experiments/batch-communication/results.json').read_text())
assert len(batch)==9
assert batch['default_five']['relu_count']==1 and batch['default_five']['cat_count']==1
assert batch['default_five']['offsets']==[0,6,12,18,24]
assert all(all(row) for row in batch['default_five']['output_storage_after'])
assert batch['below_threshold']['relu_count']==4 and batch['mixed_shapes']['relu_count']==5
assert batch['max_two']['relu_count']==3 and batch['max_two']['cat_count']==2
assert batch['require_fbgemm']['relu_count']==(1 if batch['require_fbgemm']['has_fbgemm'] else 5)
assert batch['independent_subsets']['subsets']==[['relu','relu_2']]
wait=batch['wait_schedule'];assert wait['before'].index('wait_tensor_default')<wait['before'].index('relu_default')
assert wait['after'].index('relu_default')<wait['after'].index('wait_tensor_default')<wait['after'].index('add_tensor')
assert batch['ddp_dispatch']['events']==[{'pass':'first','bucket_size_mb':7},{'pass':'second'}]
assert batch['tp_find']['count']==1 and batch['tp_find']['matches'][0]['gather_dim']==0
scatter=json.loads((ROOT/'experiments/view-scatter/results.json').read_text())
assert len(scatter)==15
assert len(scatter['nested']['canonical'])==1
assert [v['target'] for v in scatter['nested']['canonical'][0]['views']]==['aten.select.int','aten.slice.Tensor']
assert scatter['nested']['canonical'][0]['should_reinplace'] is True
assert scatter['nested']['after_core'][0]['target']=='_generalized_scatter'
assert scatter['copy_back']['after_core'][0]['target']=='_inplace_generalized_scatter'
assert 'aten.clone' not in scatter['copy_back']['final_targets'] and 'aten.copy_.default' in scatter['copy_back']['final_targets']
assert scatter['copy_back']['final']['output_input_aliases']==[[True,False]]
assert len(scatter['path_mismatch']['canonical'])==len(scatter['different_base']['canonical'])==2
assert scatter['internal']['canonical'][0]['should_reinplace'] is False
assert scatter['diagonal']['canonical'][0]['always_mutating'] and scatter['as_strided']['canonical'][0]['always_mutating']
assert scatter['diagonal']['final_targets']==['aten.clone','aten.diagonal.default','aten.copy_.default']
assert scatter['shared_inner']['final']['expected_output_aliases']==[[True,False],[False,True]]
assert scatter['shared_inner']['final']['output_aliases']==[[True,True],[True,True]]
assert scatter['shared_inner']['final']['offsets']==[0,4]
assert not scatter['direct_functional']['same_object'] and scatter['direct_inplace']['same_object']
assert scatter['copy_error']['cause']=='RuntimeError' and scatter['view_error']['error']=='IndexError'
assert scatter['empty_path_decomposition']['error']=='IndexError'
assert scatter['canonical_recompile_boundary']['error']=='SyntaxError'
reinplace=json.loads((ROOT/'experiments/reinplace-storage/results.json').read_text())
assert len(reinplace)==23
assert reinplace['copy_back']['targets']==['aten.index_put_.default']
assert all('aten.index_put.default' in reinplace[n]['targets'] for n in ['no_copy','old_view','input_view','missing_storage'])
assert 'aten.index_put_.default' in reinplace['metadata_only']['targets']
assert 'aten.index_put_.default' in reinplace['after_copy']['targets']
assert 'aten._foreach_add_.Scalar' in reinplace['foreach_full']['targets']
assert reinplace['foreach_full']['direct_fx_error']=='TypeError'
assert reinplace['foreach_full']['after_cleanup_targets']==['aten._foreach_add_.Scalar']
assert all('aten._foreach_add.Scalar' in reinplace[n]['targets'] for n in ['foreach_partial','foreach_same'])
assert reinplace['foreach_compiled']['mutated_input_idxs']==[0,1]
assert reinplace['foreach_compiled']['output_input_aliases']==[[True,False],[False,True]]
assert [reinplace[n]['decisions'][0]['only_clone_these_tensors'] for n in ['af_internal','af_shared','af_old','af_input','af_copy','v2_partial','v2_internal','v2_views']]==[[],['y'],['x'],['x','y'],[],[1],[],[0]]
assert [reinplace[n]['clone_count'] for n in ['af_internal','af_shared','af_old','af_input','af_copy','v2_partial','v2_internal','v2_views']]==[0,1,1,2,0,1,0,1]
assert reinplace['af_shared']['outputs']==[[1.,2.,3.,4.],[0.,2.,4.,6.]]
assert reinplace['mutable_list']['only_clone_these_tensors']==[] and reinplace['mutable_list']['clone_count']==0
assert reinplace['mutable_list_alias']['only_clone_these_tensors']==['xs'] and reinplace['mutable_list_alias']['clone_count']==2
assert reinplace['mutable_list']['container_type']=='immutable_list'
assert reinplace['triton_dependency']['module']=='triton' and reinplace['triton_dependency']['error']=='ModuleNotFoundError'
wrapper=json.loads((ROOT/'experiments/functional-wrapper-boundaries/results.json').read_text())
assert len(wrapper)==8
for name in ['v1_missing_decision','v2_missing_decision']:
    assert wrapper[name]['same_values_but_changed_mutation'] and wrapper[name]['clone_count']==0
    assert wrapper[name]['actual_input_after']==[1.,2.,3.,4.] and wrapper[name]['reference_input_after']==[0.,1.,2.,3.]
for name in ['v1_with_decision','v2_with_decision']:
    assert wrapper[name]['input_unchanged'] and wrapper[name]['clone_count']==1
for name,message in [('v1_retained','auto_functionalized was not removed'),('v2_retained','auto_functionalized_v2 was not removed'),('triton_retained','triton_kernel_wrapper_functional was not removed')]:
    assert wrapper[name]['error']=='AssertionError' and wrapper[name]['message']==message
assert wrapper['no_wrappers']=={'graph_unchanged':True,'return_is_none':True}
order=json.loads((ROOT/'experiments/order-entry/results.json').read_text())
assert len(order)==9
assert order['late_last_arg']['same_node_objects_and_metadata'] and order['last_arg_ready_first']['same_node_objects_and_metadata']
assert len([e for e in order['late_last_arg']['events'] if e['line']==1922])==1
assert len([e for e in order['last_arg_ready_first']['events'] if e['line']==1922])==2
assert order['incomparable_pair']['pair_before'] and not order['incomparable_pair']['pair_after']
assert order['nested_args']['dependencies']==['p','q','p','z']
assert order['forward_refresh']['before_recompile']==[1.,2.,3.,4.] and order['forward_refresh']['after_recompile']==[3.,4.,5.,6.]
assert order['cycle']['error']==order['foreign_dependency']['error']=='AssertionError'
assert order['lint_ownership']['error']=='RuntimeError'
assert order['mutation_boundary']['original_result']!=order['mutation_boundary']['changed_result']
constructor=json.loads((ROOT/'experiments/constructor-devices/results.json').read_text())
assert len(constructor)==22
for name in ['movable','output_allowed','two_constructors','gpu_boundary_two_inputs']:
    assert all(d=='cuda:0' for d in constructor[name]['constructor_kwargs_after'])
    assert constructor[name]['identity_and_metadata_unchanged']
    assert all(d=='cpu' for d in constructor[name]['metadata_devices_after'])
for name in ['output_blocked','no_target','multiple_targets','cpu_input_blocked','linked_output_blocked','repeated_input','opaque_blocked']:
    assert all(d=='cpu' for d in constructor[name]['constructor_kwargs_after'])
assert constructor['no_device_kwarg']['constructor_kwargs_after']==['absent']
assert constructor['string_device']['constructor_kwargs_after']==['cpu']
assert constructor['repeated_input']['cpu_indeg_before']['add_tensor']==2
assert all(constructor[name]['guarded'] for name in ['bool_scalar','int8_scalar','bool_singleton'])
assert not any(constructor[name]['guarded'] for name in ['uint8_scalar','int64_scalar','bool_vector','bool_gpu_value'])
assert constructor['invalid_target']['error']=='AssertionError' and constructor['tuple_metadata']['device'] is None
backend=json.loads((ROOT/'experiments/backend-registration/results.json').read_text())
assert len(backend)==10
assert all(not backend[name]['registered'] for name in ['private_default','private_no_module','private_partial_module','private_false_cpp'])
assert backend['private_complete']['registered'] and backend['private_explicit']['registered']
assert backend['private_complete']['features']==['INPLACE_BUFFERS']
assert backend['private_explicit']['cpp_selected']=='NoneType'
assert backend['private_complete']['graph_wrapper_consumer']['wrapper']=='PythonWrapper'
assert backend['private_complete']['graph_wrapper_consumer']['ops_same_object']
assert backend['private_complete']['scheduler_consumer']['same_device_cached'] and backend['private_complete']['scheduler_consumer']['different_indices_distinct']
assert not backend['private_late_module']['after_second_init'] and backend['private_late_module']['after_cache_clear_init']
assert backend['selectors']['arbitrary_key_accepted'] and backend['selectors']['torch_device_error']=='RuntimeError'
assert backend['selectors']['pre_registration_preserved'] and backend['selectors']['replacement_overwrites'] and backend['selectors']['feature_probe_with_none']
assert backend['device_ops']['unrelated_registration_prevents_lazy_builtin_import'] and backend['device_ops']['base_error']=='NotImplementedError'
assert backend['device_ops']['cuda_after_explicit_import']=='CUDADeviceOpOverrides'
assert backend['cpu_compile']['registered_classes_preserved'] and backend['cpu_compile']['input_unchanged'] and backend['cpu_compile']['output_not_aliasing_inputs']
assert backend['cpu_compile']['output']==[[0.,0.,0.],[2.,4.,6.]]
assert [e['kind'] for e in backend['cpu_compile']['events']]==['wrapper','scheduling','scheduling','group','flush','codegen_node','flush','kernel_call']
links=0
for p in ROOT.rglob('*.md'):
    markdown=re.sub(r'```[^\n]*\n.*?\n```','',p.read_text(),flags=re.S)
    markdown=re.sub(r'`[^`\n]*`','',markdown)
    for target in re.findall(r'\]\(([^)]+)\)',markdown):
        if '://' in target:continue
        m=re.match(r'(.+):(\d+)$',target)
        target_path=m[1] if m else target
        f=Path(target_path) if target_path.startswith('/') else p.parent/target_path
        assert f.exists(),(p,target)
        if m:assert 1<=int(m[2])<=len(f.read_text().splitlines()),(p,target)
        links+=1
for p in [*ROOT.joinpath('experiments').glob('*.py'),*ROOT.joinpath('assets').glob('*.py')]:
    ast.parse(p.read_text())
r=json.loads((ROOT/'experiments/stages/results.json').read_text())
assert len(r)==7
assert r['view']['aliases_input0']==[True]
assert r['clone_output']['aliases_input0']==[False]
assert r['reduction_large']['output']==[[2080.,6176.]]
assert all(c['generated_kernels']==(0 if name=='view' else 1) for name,c in r.items())
stages=ROOT/'experiments/stages'
assert 'aten.clone.default' in (stages/'clone_internal_before_post.txt').read_text()
assert 'aten.clone.default' not in (stages/'clone_internal_after_post.txt').read_text()
assert 'aten.clone.default' in (stages/'clone_output_after_post.txt').read_text()
assert 'data=Pointwise(' in (stages/'reduction_lowered.txt').read_text()
assert 'data=Reduction(' in (stages/'reduction_large_lowered.txt').read_text()
assert 'FusedSchedulerNode' not in (stages/'multiple_outputs_before_fusion.txt').read_text()
assert 'FusedSchedulerNode' in (stages/'multiple_outputs_after_fusion.txt').read_text()
assert 'buf1 = buf0; del buf0  # reuse' in (stages/'mm_output.py').read_text()
p=json.loads((ROOT/'experiments/properties.json').read_text())
assert p['before']==p['after_rewrite_before_update']
assert p['after_update']=={'relu':[2],'mul':[2]}
assert p['actual']==[-6.,12.] and p['input_unchanged']
rewrite=json.loads((ROOT/'experiments/rewrites-mutation/results.json').read_text())
assert len(rewrite)==6
assert rewrite['addmm_tensor']['after_targets']==['aten.addmm.default']
assert rewrite['addmm_scalar']['after_targets']==['aten.mm.default','aten.add.Tensor']
assert rewrite['copy_back']['after_targets']==['aten.index_put_.default']
assert rewrite['copy_back']['input0_after']==[0.,9.,2.,3.] and rewrite['copy_back']['output_alias_input0']==[True]
assert rewrite['input_only']['after_targets']==['aten.index_put.default']
assert rewrite['input_only']['input0_after']==[0.,1.,2.,3.]
assert rewrite['old_reader']['output'][0]==[2.,3.,4.,5.]
assert 'aten.copy_.default' in rewrite['old_reader']['after_targets']
final=json.loads((ROOT/'experiments/finalization.json').read_text())
assert final['sort']['after']==['x_1','add','mul','output'] and final['cycle']=='AssertionError'
assert final['movable']['constructor_after']=='cuda:0'
assert all(final[n]['constructor_after']=='cpu' for n in ['output_cpu','multiple_devices','boolean_index'])
interp=json.loads((ROOT/'experiments/interpretation.json').read_text())
assert interp['input_stride']==[1,2] and interp['constants']==['_tensor_constant0']
assert [(e['name'],e['operations_so_far']) for e in interp['events'] if e['name'] in ['add','mul']]==[('add',0),('mul',1)]
assert interp['events'][-1]['ir_chain']==['NoneType'] and interp['outputs']==1
storage=json.loads((ROOT/'experiments/expressions-storage/results.json').read_text())
assert len(storage)==8
assert storage['expand']['strides']==[[0,1]] and storage['expand']['generated_kernels']==0
assert storage['expand']['output_input_aliases']==[[True]]
assert storage['permute']['strides']==[[1,3]]
assert storage['reshape_copy']['outputs']==[[0.,3.,1.,4.,2.,5.]]
assert storage['reshape_copy']['output_input_aliases']==[[False]]
assert [(e['name'],e['operations']) for e in storage['reuse_exp']['events'] if e['name'] in ['exp','add','mul']]==[('exp',1),('add',2),('mul',3)]
reuse=(ROOT/'experiments/expressions-storage/reuse_exp_output.py').read_text()
assert 'buf0 = empty' not in reuse and 'buf1 = empty' in reuse and 'buf2 = empty' in reuse
for case in ['copy_old','mutate_only','dead_old']:
    assert storage[case]['mutated_input_idxs']==[0]
    assert storage[case]['inputs_after'][0]==[9.,9.,9.,9.]
assert storage['copy_old']['outputs'][0]==[1.,2.,3.,4.]
assert "WeakDep(name='buf0', mutating_buf='buf1')" in (ROOT/'experiments/expressions-storage/copy_old_dependencies.txt').read_text()
assert 'op0:' in (ROOT/'experiments/expressions-storage/dead_old_dependencies.txt').read_text()
assert 'op0:' not in (ROOT/'experiments/expressions-storage/dead_old_before_fusion.txt').read_text()
assert 'WeakDep' not in (ROOT/'experiments/expressions-storage/dead_old_before_fusion.txt').read_text()
special=json.loads((ROOT/'experiments/special-ops/results.json').read_text())
assert special['fallback_forbidden']['error']=='MissingOperatorWithoutDecomp'
assert special['fallback']['runs'][0]['outputs']==[[0.,1.,4.],[-4.,0.,6.]]
assert special['cond']['runs'][0]['outputs']==[[-1.,1.,4.]]
assert special['cond']['runs'][1]['outputs']==[[-4.,0.,6.]]
assert special['while']['runs'][0]['outputs']==[3,[1.,3.,6.]]
assert special['while']['runs'][1]['output_input_aliases']==[[True,False],[False,True]]
assert special['while']['lowerings'][-1]['never_reuse_buffers']==['buf1','buf2']
boundary=json.loads((ROOT/'experiments/entry_boundary.json').read_text())
assert boundary['error']=='AssertionError' and 'aten.t.default' in boundary['message']
fusion=json.loads((ROOT/'experiments/fusion-lifetime/results.json').read_text())
assert len(fusion['fusion_allowed']['fusion']['after'])==1
assert fusion['fusion_allowed']['fusion']['after'][0]['kind']=='FusedSchedulerNode'
assert len(fusion['fusion_limited']['fusion']['after'])==2
assert fusion['fusion_allowed']['outputs']==fusion['fusion_limited']['outputs']
assert fusion['reuse_allowed']['last_usage']==fusion['reuse_disabled']['last_usage']
assert 'buf1 = buf0; del buf0' in (ROOT/'experiments/fusion-lifetime/reuse_allowed_output.py').read_text()
assert 'buf1 = empty_strided_cpu' in (ROOT/'experiments/fusion-lifetime/reuse_disabled_output.py').read_text()
cache=json.loads((ROOT/'experiments/cache-lifecycle/results.json').read_text())['workers']
assert len({row['pid'] for row in cache.values()})==5
assert cache['cold']['calls']=={'lowering':1,'post_compile':1}
assert cache['warm']['calls']=={'lowering':0,'post_compile':1}
assert cache['new_shape']['calls']=={'lowering':0,'post_compile':1}
assert cache['cold']['cache_key']==cache['warm']['cache_key']==cache['new_shape']['cache_key']
assert cache['warm']['output'][0][0]==202. and cache['new_shape']['shape']==[3,4]
assert cache['new_dtype']['calls']['lowering']==1 and cache['new_dtype']['cache_key']!=cache['cold']['cache_key']
assert cache['bypass']['counters']['fxgraph_cache_bypass']==1 and cache['bypass']['calls']['lowering']==1
async_result=json.loads((ROOT/'experiments/async_loading.json').read_text())
assert async_result['before_is_future'] and not async_result['after_is_future'] and async_result['callable_after_wait']
assert async_result['actual']==[12.]
initial=json.loads((ROOT/'experiments/initial_properties.json').read_text())
assert len(initial)==6
assert initial['new_mode']['target']=='aten.reshape.default' and initial['new_mode']['same_node'] and initial['new_mode']['forward_still_old_before_recompile']
assert initial['reuse']['same_mode'] and not initial['reuse']['allow_after']
assert initial['mixed_rejected']['error']=='AssertionError' and initial['mode_conflict']['error']=='AssertionError'
assert initial['mixed_allowed']['same_mode'] and not initial['mixed_allowed']['allow_after']
assert initial['ambient']['same_mode'] and not initial['ambient']['allow_after']
coverage=json.loads((ROOT/'maintenance/leaf-coverage.json').read_text())
for item in coverage:
    source=Path(item['file']).read_text().splitlines()
    segment='\n'.join(source[item['start']-1:item['end']])
    assert hashlib.sha256(segment.encode()).hexdigest()==item['sha256'],item
    assert '```python\n'+segment+'\n```' in (ROOT/item['article']).read_text(),item
for n in nodes.values():
    if n['status']=='叶子·逐行完成':
        assert not any(c['parent']==n['id'] for c in nodes.values())
        assert any(i['article']==n['path'] for i in coverage)
incremental=json.loads((ROOT/'experiments/incremental_properties.json').read_text())
assert incremental['shape_change']['a_shape']==incremental['shape_change']['b_shape']==[2]
assert incremental['equal_attributes']['same_metadata_object'] and not incremental['equal_attributes']['new_signature_saved']
assert incremental['equal_attributes']['neg_calls_total']>incremental['equal_attributes']['neg_calls_first']>0
assert incremental['missing_input_metadata']['same_metadata_object'] and not incremental['missing_input_metadata']['signature_saved']
assert incremental['dtype_only']['metadata_dtype']=='torch.float32' and incremental['dtype_only']['actual_dtype']=='torch.float64'
assert incremental['alias_to_clone']['before_aliases_input'] and not incremental['alias_to_clone']['after_aliases_input']
cleanup=json.loads((ROOT/'experiments/cleanup-noops/results.json').read_text())
assert cleanup['pure']['before_order']!=cleanup['pure']['after_order']
assert cleanup['copy_tail']['before_order']==cleanup['copy_tail']['after_order']
assert cleanup['other_mutation']['regions']['sin']==0 and cleanup['other_mutation']['regions']['add']==1
assert 'sin' not in cleanup['dce']['after_order'] and 'copy_' in cleanup['dce']['after_order']
assert cleanup['clone_internal']['after_targets']==['aten.relu.default']
assert cleanup['clone_output']['after_targets']==['aten.clone.default']
assert cleanup['cat_internal']['after_targets']==['aten.relu.default'] and cleanup['cat_output']['after_targets']==['aten.cat.default']
assert cleanup['primitive_same_dtype']['after_targets']==['aten.relu.default']
assert 'prims.convert_element_type.default' in cleanup['primitive_changes_dtype']['after_targets']
assert cleanup['alias_tuple']['identity_before'] is False and cleanup['alias_tuple']['identity_after'] is True
assert cleanup['alias_tuple']['storage_alias_before'] and cleanup['alias_tuple']['storage_alias_after']
assert cleanup['alias_single_output']['after_targets']==['aten.alias.default'] and not cleanup['alias_single_output']['identity_after']
device_wrapper=json.loads((ROOT/'experiments/wrapper-devices/results.json').read_text())
assert len(device_wrapper)==16
assert device_wrapper['stream_protocol']['stream_queries_two_invocations']==2
assert [x[0] for x in device_wrapper['stream_protocol']['first_events']]==['guard_enter','set_device','stream','a','b','guard_exit']
assert device_wrapper['stream_cache_keys']['assignments']==2 and device_wrapper['stream_cache_keys']['import_count']==1
assert device_wrapper['autotune_sections']['main_launches']==2 and device_wrapper['autotune_sections']['example_launches']==1
assert device_wrapper['argument_types_rejected']['error']=='AssertionError'
assert device_wrapper['aot_mismatch']['error']=='AssertionError'
assert 'AOTICudaGuard' in device_wrapper['cpp_jit_first']['source']
assert 'set_index(3)' in device_wrapper['cpp_jit_next']['source']
assert 'AOTICudaStreamGuard' in device_wrapper['aot_first']['source']
assert not device_wrapper['aot_same']['source'].strip()
assert device_wrapper['cpu_compile']['args_cleared'] and device_wrapper['cpu_compile']['input_tensor_unchanged']
assert device_wrapper['cpu_compile']['output']==[[1.,2.,3.],[4.,5.,6.]]
assert all(x['error']=='AssertionError' for x in device_wrapper['cpu_compile']['layout_rejections'].values())
context=json.loads((ROOT/'experiments/graph-context/results.json').read_text())
assert len(context)==13
assert context['fresh']['runtime_asserts_frozen'] and not context['fresh']['reuse_shape_env']
assert context['assert_snapshot']['copied_dict'] and context['assert_snapshot']['shared_value_lists']
assert context['assert_snapshot']['source_keys_survive_pop'] and context['assert_snapshot']['new_unknown_assert_error']=='AssertionError'
assert context['reuse_symbols']['same_expressions']
assert context['fresh_symbols']['same_size_shared_symbol'] and context['fresh_symbols']['static_size_stride']==[[2,3],[1,2]]
assert context['device_registration']['types']==['cpu','cuda','xpu'] and context['device_registration']['indices']==[0,1]
assert context['device_registration']['first_node_by_device']=={'cuda:0':'x_1','cuda:1':'x_1','xpu:0':'add'}
assert context['mixed_devices']['error']=='AssertionError'
assert context['constant_sharing']['same_devices'] and context['constant_sharing']['same_constants'] and context['constant_sharing']['separate_operations']
assert context['subgraph']['same_shape_env'] and context['subgraph']['separate_devices'] and context['subgraph']['qualified_name']=='outer_branch_buf0'
assert all(context['scope_restore'].values()) and context['setter_timing']['installed_before_enter']
assert all(context['thread_local']['workers'].values()) and context['thread_local']['main_kept_own_graph']
assert context['feature_cache']['provider_calls']==['cpu','cpu'] and context['feature_cache']['invalid_feature']=='AssertionError'
assert context['lowering']['inputs']==['x_1','y_1'] and context['lowering']['first_device_node']=='x_1' and context['lowering']['operations']==1
legality=json.loads((ROOT/'experiments/fusion-legality/results.json').read_text())
assert len(legality['protocol'])==18 and len(legality['dependencies'])==9 and len(legality['simd'])==5 and len(legality['cpu'])==4
for name in ['horizontal','vertical','peak_distance_64','aggressive_no_shared']:
    assert legality['protocol'][name]['allowed']
for name in ['reverse','different_device','max_size','explicit_no_fuse','template_as_consumer','template_reduction_epilogue','offset_mismatch','star_unmatched','intermediate','peak_distance_65','no_shared','aggressive_reduction','same_node']:
    assert not legality['protocol'][name]['allowed'] and not legality['protocol'][name]['backend_calls']
assert legality['protocol']['backend_veto']['backend_calls']==['horizontal'] and not legality['protocol']['backend_veto']['allowed']
assert legality['dependencies']['broadcast_prefix']['allowed'] and legality['dependencies']['mode_memory_early_return']['allowed']
assert not legality['dependencies']['shift']['allowed'] and not legality['dependencies']['indirect']['allowed']
assert legality['dependencies']['weak']=={'same_index':True,'different_index':False}
assert legality['cycles']['new_path'] and not legality['cycles']['no_new_path']
assert legality['simd']['pointwise_equal']['allowed'] and not legality['simd']['pointwise_mismatch']['allowed']
assert legality['simd']['split_scan_reason_only']['allowed'] and legality['simd']['split_scan_reason_only']['reasons']==['Split scan cannot fuse with reductions']
assert legality['cuda_cpp_horizontal_boundary']['error']=='NotImplementedError'
assert legality['base_hooks']=={'vertical':'NotImplementedError','horizontal':'NotImplementedError'}
assert legality['cpu_benchmark_bypass']=={'allowed':True,'benchmark_called':False}
for direction in ['horizontal','vertical']:
    allowed=legality['cpu'][direction+'_allow'];vetoed=legality['cpu'][direction+'_veto']
    assert len(allowed['after'])==1 and len(vetoed['after'])==2
    assert allowed['generated_kernel_count']==1 and vetoed['generated_kernel_count']==2
    assert allowed['outputs']==vetoed['outputs']
    assert all(row['input_preserved'] and row['outputs_do_not_alias_input'] for row in [allowed,vetoed])
    assert len(allowed['calls'])==2 and len(vetoed['calls'])==1
    assert all(call['hook']==direction for call in allowed['calls']+vetoed['calls'])
elements=json.loads((ROOT/'experiments/element-codegen/results.json').read_text())
assert len(elements['cpu'])==4 and len(elements['protocol'])==7
for label,row in elements['cpu'].items():
    assert row['input_preserved'] and not row['output_input_aliases']
    assert row['generated_kernel_count']==1
    assert row['comparison']['eager_match']==(label!='roundtrip_fp16')
shared=elements['cpu']['shared_exp'];exp_events=[e for e in shared['cse'] if 'std::exp(' in e['expr']]
assert len(exp_events)==1 and not exp_events[0]['hit_before']
assert len(shared['bodies'])==3 and all(row['kind']=='CppKernel' for row in shared['bodies'])
assert "ops.load('buf0'" in shared['bodies'][1]['loop_graph']
assert shared['bodies'][0]['loads']==shared['bodies'][1]['loads']==shared['bodies'][2]['loads']
assert 'out_ptr0[' not in (ROOT/'experiments/element-codegen/shared_exp_output.py').read_text()
assert elements['cpu']['strided_broadcast']['outputs']==[[[10.,22.,34.],[11.,23.,35.]]]
assert elements['cpu']['strided_broadcast']['output_strides']==[[1,2]]
rounding=elements['cpu']['roundtrip_fp16']['comparison']
assert rounding['equals_fp32_input'] and rounding['max_abs_diff']>0.0009 and rounding['eager_output'][0][0]==0.0999755859375
assert 'c10::convert<half>' in (ROOT/'experiments/element-codegen/fp16_output_output.py').read_text()
p=elements['protocol'];assert p['cse_scope']['same_variable'] and p['cse_scope']['use_count']==2
assert p['cse_scope']['invalidated']==['buf1'] and p['cse_scope']['retained_store']==['buf0']
assert p['cse_scope']['clone_shared_store'] and p['cse_scope']['clone_next_name']=='tmp2'
assert p['args']['defs']==['float* in_out_ptr0','const int64_t ks0'] and p['args']['calls']==['buf0','s0']
assert p['workspace']['total_bytes']==48 and p['workspace']['second']==['ws_ptr',16] and p['workspace']['zero_fill']
assert 'Workspace not supported on CPU' in p['workspace']['cpp_error'] and p['removed_arg']['error']=='AssertionError'
assert p['proxy']['forwarded'] and p['proxy']['removed_store_skipped'] and p['proxy']['must_keep']==['buf0']
assert p['triton_dtype']['min_elem_per_thread']==4 and p['triton_dtype']['cross_fp8_error']=='AssertionError'
assert p['triton_masks']['masks']==['rmask','xmask']
external=json.loads((ROOT/'experiments/external-fallback/results.json').read_text())
assert len(external['cpu'])==5 and len(external['boundaries'])==5 and not external['config']['implicit_fallbacks']
for label,row in external['cpu'].items():
    assert len(row['runs'])==2 and all(run['runtime_calls']==1 for run in row['runs'])
    assert sum(value for key,value in row['compile_counts'].items() if key.endswith('_cpu'))==0
    assert sum(value for key,value in row['compile_counts'].items() if key.endswith('_fake'))==2
    assert len([node for node in row['operations'] if node['kind']=='FallbackKernel'])==1
packed=next(n for n in external['cpu']['packed']['operations'] if n['kind']=='FallbackKernel')
assert len(packed['tensor_inputs'])==2 and packed['non_tensor_args']==['0','3.0','True']
assert packed['outputs'][1][0]['indices']==[['tuple',1],['list',0]]
assert packed['outputs'][1][1]['indices']==[['tuple',1],['list',1]]
assert external['cpu']['packed']['runs'][0]['outputs'][0]==[[5.,7.,10.],[8.,6.,9.]]
assert all(not any(alias_row) for alias_row in external['cpu']['packed']['runs'][0]['output_input_aliases'])
for label in ['contiguous','mutation_copy']:
    assert external['cpu'][label]['input_layouts'][0]['stride']==[1,2]
    assert all(run['runtime_events'][0]['stride']==[3,1] for run in external['cpu'][label]['runs'])
for label in ['mutation','mutation_copy']:
    node=next(n for n in external['cpu'][label]['operations'] if n['kind']=='FallbackKernel')
    assert node['side_effects'] and node['mutation_names']==[] and len(node['mutation_outputs'])==1
    assert 'x_1' in external['cpu'][label]['mutated_buffers']
assert external['cpu']['mutation_copy']['runs'][0]['inputs_after']==[[[3.,5.,7.],[4.,6.,8.]]]
assert all(run['output_input_aliases']==[[True]] for run in external['cpu']['view']['runs'])
boundaries=external['boundaries'];assert boundaries['unsupported_mutable_alias']['error']=='LoweringException'
assert "NYI: Can't generate FallbackKernel" in boundaries['unsupported_mutable_alias']['message']
assert boundaries['fake_is_not_runtime']['dispatch_keys']=={'CPU':True,'CUDA':False,'Meta':True}
assert boundaries['find_device']['first_input_wins']=='cpu' and boundaries['find_device']['no_tensor_sequence_error']=='IndexError'
assert boundaries['callable']['error']=='RuntimeError' and boundaries['decomp']['error']=='AssertionError'
integration=json.loads((ROOT/'experiments/template-integration/results.json').read_text())
assert len(integration['cpu'])==7
for row in integration['cpu'].values():
    assert row['input_preserved'] and row['captured_weight_preserved'] and not row['output_aliases_input']
    assert row['benchmark_fn_calls']==0
cpp=integration['cpu']['mm_cpp'];assert cpp['selectors'][0]['result_kind']=='CppTemplateBuffer'
assert cpp['selectors'][0]['choices'][0]['wrapped_kind']=='CppTemplateCaller'
assert any(node['kind']=='CppTemplateBuffer' for node in cpp['operations'])
assert cpp['constants']['constant0']['shape']==[1,16,48]
assert cpp['output_shapes']==[[8,16]]
assert integration['cpu']['mm_external']['selectors'][0]['result_kind']=='ExternKernelOut'
assert not integration['cpu']['mm_empty_fallback']['selectors']
assert not integration['cpu']['bmm_decomposed']['selectors']
assert integration['cpu']['bmm_decomposed']['operations'][0]['kind']=='ComputedBuffer'
assert integration['cpu']['bmm_empty_backends']['selectors'][0]['choices'][0]['kind']=='ExternKernelCaller'
assert not integration['cpu']['bmm_empty_backends']['config']['autotune_fallback_to_aten']
assert integration['cpu']['foreach_cpu']['lists']=={}
assert len(integration['lowering_only']['foreach_feature']['lists'])==1
assert integration['boundaries']['mm_empty_rejected']['error']=='LoweringException'
assert 'NoValidChoicesError' in integration['boundaries']['mm_empty_rejected']['message']
assert integration['boundaries']['triton_device_gate']['gates']=={'cpu':False,'cuda:0':True,'xpu:0':False,'privateuseone:0':False}
user_kernel=json.loads((ROOT/'experiments/template-integration/user-kernel-protocol.json').read_text())
assert user_kernel['return_keys']==['inp','out'] and user_kernel['same_tensorboxes']
assert user_kernel['ordered_args']==['inp','out','scale'] and user_kernel['constant_overrode_scale']==2.
assert user_kernel['mutated_buffers']==['y_1'] and user_kernel['mutation_outputs']==[['y_1']]
assert user_kernel['wrapper_events'][-1]['constexpr_indices']==[2] and user_kernel['wrapper_events'][-1]['args']==['x_1','y_1',2.]
assert user_kernel['no_tensor_error']=='AssertionError'
selection=json.loads((ROOT/'experiments/algorithm-selection/real-cpu.json').read_text())
cold,warm=selection['cold'],selection['warm']
assert cold['benchmark_fn_calls']==1 and warm['benchmark_fn_calls']==0
assert len(cold['timings_ms'])==2 and cold['timings_ms']==warm['timings_ms']
assert cold['candidate_hashes']==warm['candidate_hashes'] and cold['cache_path']==warm['cache_path']
assert len(cold['precompile_calls'])==1 and warm['precompile_calls']==[]
assert cold['output_sample']!=warm['output_sample']
for row in [cold,warm]:
    assert row['verify_enabled'] and row['input_preserved'] and row['captured_weight_preserved'] and not row['output_aliases_input']
    assert row['output_shape']==[16,64]
    assert row['selected_choice']==min(row['timings_ms'],key=row['timings_ms'].get)
protocol=json.loads((ROOT/'experiments/algorithm-selection/protocol.json').read_text())
assert len(protocol)==11
assert protocol['filtered_single']=={'output':'a','lookup_calls':0}
assert protocol['empty_filtered']['error']=='NoValidChoicesError' and protocol['all_failed']['error']=='NoValidChoicesError'
assert protocol['empty_timing_map']['output']=='a' and protocol['partial_second_only']['output']=='a'
assert protocol['full_rank_map']['output']=='b'
assert protocol['single_cuda_exception']['lookup_calls']==1
assert protocol['deferred_once']['kind']=='MultiTemplateBuffer' and protocol['deferred_once']['lookup_before_property']==0 and protocol['deferred_once']['lookup_after_two_reads']==1
assert protocol['cpu_disables_defer']['lookup_calls']==protocol['input_generator_disables_defer']['lookup_calls']==1
tasks=json.loads((ROOT/'experiments/compile-tasks.json').read_text())
assert tasks['cpu']['1']['compile_submissions']==0 and tasks['cpu']['2']['compile_submissions']==1
assert tasks['cpu']['1']['before']=='builtin_function_or_method' and tasks['cpu']['2']['before']=='LambdaFuture'
for threads,row in tasks['cpu'].items():
    assert row['output']==[12.] and row['after']=='builtin_function_or_method'
    assert row['task']['pid']==tasks['main']['pid']
    assert (row['task']['thread']==tasks['main']['thread'])==(threads=='1')
assert tasks['process']['worker']['pid']!=tasks['main']['pid']
assert tasks['process']['wait_result']['pid']!=tasks['main']['pid']
assert tasks['process']['ordinary_job_error']=='SubprocException' and not tasks['process']['device_compilation']
tp=tasks['protocol']
assert tp['partial_failure']=={'error':'ValueError','first':17,'bad_same':True,'later_same':True}
assert tp['shallow_scope']['nested_preserved'] and tp['threads1_wait_preserves_injected_future']
assert tp['broken_pool']=={'error':'RuntimeError','cause':'BrokenProcessPool'}
assert tp['triton_False']['precompile_flags']==[{}]
assert tp['triton_True']['precompile_flags']==[{'warm_cache_only':True},{}]
binary_cache=json.loads((ROOT/'experiments/source-binary-cache/results.json').read_text())
bn=binary_cache['native']
assert bn['cold']['key']==bn['warm']['key']!=bn['variant']['key']
assert [bn[k]['build_calls'] for k in ['cold','warm','variant']]==[1,0,1]
assert [bn[k]['value'] for k in ['cold','warm','variant']]==[12,12,14]
assert sum(bn[k]['build_calls'] for k in ['race_a','race_b'])==1
assert bn['race_a']['key']==bn['race_b']['key']==bn['cold']['key']
for label,row in bn.items():
    assert row['same_getter'] and row['same_library'] and row['source_exists'] and row['binary_exists']
    assert row['build_calls']==sum(row['binary_path'] in command for command in row['commands'])
    assert row['all_build_calls_including_isa_probes']==len(row['commands'])
assert all(binary_cache['keys'].values()) and len(binary_cache['keys'])==6
for row in binary_cache['device_protocol'].values():
    assert row['fixtures_only'] and row['extra_args_same_key'] and row['memory_then_disk_skip']
    assert row['compiler_calls']==1 and row['changed_extension_changes_key']
loading=json.loads((ROOT/'experiments/jit-aot-loading/results.json').read_text())
lj=loading['jit'];assert lj['output']==[[10.,21.,32.],[13.,24.,35.]]
assert lj['input_preserved'] and lj['constant_preserved'] and lj['args_consumed'] and not lj['output_aliases_input']
assert lj['attrs']==['_tensor_constant0'] and lj['module_has_call']
assert [e['phase'] for e in lj['events']]==['load_request','module_executed','load_return']
assert lj['events'][1]['constant_placeholders']=={'_tensor_constant0':True}
assert lj['events'][2]['attrs_present']
lp=loading['module_protocol'];assert lp['attrs_cache_hit']['value']==11 and lp['attrs_cache_hit']['same_module'] and lp['attrs_cache_hit']['original_constant_retained']
assert not lp['attrs_cache_hit']['has_reload_callback'] and not lp['attrs_cache_hit']['seen_constant_during_exec']
assert lp['plain_reload']['new_module'] and lp['plain_reload']['same_path']
assert lp['syntax']['error']=='RuntimeError' and lp['exec']['error']=='ValueError'
assert lp['aot_loader_fixture']['fixtures_only'] and lp['aot_loader_fixture']['unsupported_error']=='Unsupported device privateuseone:0'
la=loading['aot'];assert la['compiled'] and la['loaded'] and la['exists'] and la['constant_names_to_fqns']=={'weight':'weight'}
assert la['input_preserved'] and not la['output_aliases_input'] and la['output']==lj['output']
lar=json.loads((ROOT/'experiments/jit-aot-loading/aot-runtime.json').read_text())
assert not lar['compiled_in_this_process'] and lar['path']==la['path'] and lar['constant_names_to_fqns']=={'weight':'weight'}
assert len(lar['runs'])==2 and lar['runs'][0]['output']!=lar['runs'][1]['output']
for run in lar['runs']:
    assert run['input_preserved'] and not run['output_aliases_input'] and run['output_type']=='tuple'
adapt=json.loads((ROOT/'experiments/call-adaptation/results.json').read_text())
for row in adapt['cpu'].values():
    assert row['input_mod16']==4 and row['args_consumed']
    assert row['events'][0]['checks']==[] and not row['events'][0]['boxed_after']
    assert 'cpu device' in row['events'][0]['disabled_reason'] and row['events'][0]['same_callable']
assert adapt['cpu']['pure']['input_preserved'] and not adapt['cpu']['pure']['output_aliases_input']
assert adapt['cpu']['pure']['output']==[[3.,5.,7.],[9.,11.,13.]]
assert not adapt['cpu']['mutation']['input_preserved'] and adapt['cpu']['mutation']['output_aliases_input']
assert adapt['cpu']['mutation']['output']==[[4.,5.,6.],[7.,8.,9.]]
ah=adapt['alignment']['real_cpu_helper'];assert ah['checks_for_actual_cpu']==[] and ah['before_mod16']==4
assert ah['seen']==[{'shape':[4,3],'stride':[1,4],'aligned':True,'same_tensor':False}]
assert ah['replaced_list_item'] and ah['original_preserved'] and ah['empty_checks_same_callable'] and ah['aligned_tensor_same_object']
assert ah['filtered_static']==[0]
assert all(adapt['alignment']['explicit_helper_mutation'].values())
am=adapt['alignment']['device_metadata'];assert am['nonstatic']==[0] and am['static']==[] and am['assume_aligned']==[0,1] and am['fake_only']
assert am['is_gpu']=={'cpu':False,'cuda':True,'xpu':True,'privateuseone':False}
ap=adapt['policy'];assert ap['stride_context']['output_strides']==[[3,1],None]
for key in ['cached_disabled','failed_gate']:
    assert not ap[key]['boxed'] and ap[key]['factory_calls']==[] and ap[key]['output_correct']
assert ap['enabled']['boxed'] and ap['enabled']['factory_calls']==[{'device_index':2,'static_input_idxs':[]}]
assert ap['backward_notice']['events']==['backward','model'] and not ap['backward_notice']['boxed']
assert all(ap['mutation_gate'].values())
ad=adapt['deferred_protocol']
for key in ['trees_False','trees_True']:
    assert ad[key]['factory_calls']==1 and ad[key]['outputs']==[4.,9.] and ad[key]['capture_fixture']
assert ad['capture_error']['factory_attempts']==2 and ad['capture_error']['error']=='fixture capture failed'
assert ad['tree_integer_keys']['factory_keys']==[7,8] and ad['tree_integer_keys']['outputs']==[8.,8.,9.]
registry_leaf=json.loads((ROOT/'experiments/backend-registry-leaf/results.json').read_text())
assert len(registry_leaf)==6
rd=registry_leaf['dynamic_defaults'];assert rd['getter_does_not_init'] and rd['same_factory']
assert rd['cpp_kind']=='CppScheduling' and rd['halide_kind']=='HalideScheduling' and rd['invalid_config']=='KeyError'
assert rd['xpu_cpp_slot_is_none_type'] and rd['no_kernel_compilation']
rp=registry_leaf['feature_probes'];assert rp['initial_probe_instances']==3 and rp['returns_same_feature_object']
assert rp['membership_true'] and not rp['membership_false'] and rp['cpu_index_string_error']=='TypeError'
assert rp['string_exact_key_features']==['SCAN'] and rp['device_object_type_features']==['INPLACE_BUFFERS']
assert rp['bad_feature']=='AssertionError' and rp['missing_scheduling']=='TypeError'
ro=registry_leaf['ops_import_cache'];assert ro['fresh_import_keys']==['cuda','xpu'] and ro['clear_after_import_returns_none'] and ro['explicit_overwrite_identity']
assert len(ro['base_methods_reject'])==4
re=registry_leaf['private_exception'];assert re['getter_fixture_error']=='ValueError' and re['failed_init_cached_entries']==0
assert re['defaults_remain_after_error']==['cpu','cuda','xpu'] and re['retry_registered'] and re['cache_after_retry'][-1]==1
rf=registry_leaf['private_false_value'];assert rf['both_unregistered']
assert rf['false_value_events'][-1]=={'name':'CppWrapperCodeGen','truth':False}
assert rf['none_value_events'][-1]=={'name':'CppWrapperCodeGen','error':'RuntimeError'}
rr=registry_leaf['raw_slots'];assert rr['none_scheduling_replaced_by_default'] and rr['invalid_implementation_slots_stored'] and rr['explicit_none_slot'] and rr['omitted_cpp_slot_is_none_type'] and rr['missing_getters_none']
assert rr['arbitrary_feature_query_error']=='RuntimeError'
dispatch=json.loads((ROOT/'experiments/backend-dispatch/results.json').read_text())
dc=dispatch['cpu'];assert dc['backend_instances']==1 and dc['input_preserved'] and not dc['output_aliases_input']
assert dc['output']==[[0.,0.,2.],[4.,6.,8.]]
assert sum(e['kind']=='create' for e in dc['events'])==1 and any(e.get('hit') for e in dc['events'])
dd=dispatch['decode'];assert dd['worker_lookups']==['cuda','xpu','privateuseone'] and dd['specified_same_object']
assert dd['values']['cuda']=='cuda:2' and dd['values']['privateuseone']=='privateuseone:2' and dd['default']=='cpu'
dp=dispatch['creation_protocol'];assert all(dp['cache'].values())
assert dp['unnormalized']['error']=='AssertionError' and dp['unnormalized']['added']==[]
assert dp['missing']['error']=='RuntimeError' and dp['missing']['added']==['meta']
assert 'too old' in dp['old_cuda']['message'] and dp['old_cuda']['property_calls']==1
assert 'working triton' in dp['no_triton_cuda']['message'] and 'working triton' in dp['no_triton_xpu']['message']
assert dp['no_triton_xpu']['property_calls']==0
assert dp['cpu_without_triton']['kind']==dp['private_without_triton']['kind']=='Backend'
assert dp['constructor_failure']=={'attempts':2,'cache_size':0,'device_record_calls':['cpu','cpu']}
db=dispatch['base'];assert len(db['unimplemented'])==9 and not db['features'] and not db['ready_to_flush'] and db['fusion_priority']==0 and db['scheduler_constructor_error']=='TypeError'
assert db['merge_branches']==['ordinary-merge','foreach-merge','foreach-merge']
dco=dispatch['combined_protocol'];assert dco['choices']=={'ordinary':'triton','cuda':'cuda','rocm':'rocm','both':'cuda'}
assert dco['nonempty_epilogue_rejects']==['cuda','rocm']
assert len(dco['forwarded'])==9 and all(v=='triton.'+k for k,v in dco['forwarded'].items())
assert dco['vertical_short']==['cuda.vertical'] and dco['vertical_fallback']==['cuda.vertical','triton.vertical']
assert dco['horizontal_cuda']==['cuda.horizontal'] and dco['horizontal_rocm']==['triton.horizontal']
assert dco['real_cuda_horizontal']=='NotImplementedError' and not dco['ready_to_flush_inherited'] and dco['priority_inherited']==0
report={'backend_dispatch_cpu_graphs':1,'backend_dispatch_protocol':'device properties/workers/delegates fixtures; no GPU compilation','backend_registry_leaf_workers':len(registry_leaf),'call_adaptation_native_cpu_graphs':len(adapt['cpu']),'call_adaptation_gpu':'metadata/factory fixtures, no GPU capture or replay','call_adaptation_protocol_sections':len(adapt['alignment'])+len(ap)+len(ad),'jit_loading':'actual CPU graph with constant attrs','aot_cpu_loading':'real so, native runner, separate-process aot_load twice','loading_protocol_cases':len(lp),'source_binary_native_cpu_processes':len(bn),'source_binary_key_cases':6,'source_binary_device_protocol':'CUDA/ROCm fake compiler only','compile_tasks_native_cpu_paths':2,'compile_tasks_process_pool':'actual ordinary Python jobs; no device compiler','compile_tasks_protocol_cases':len(tp),'algorithm_selection_native_cpu_processes':2,'algorithm_selection_protocol_cases':len(protocol)-1,'algorithm_timing_cache':'cold benchmark, warm hit; different model values','template_integration_cpu_graphs':len(integration['cpu']),'real_cpp_template':'compiled and executed; single choice, no benchmark','user_kernel_protocol':'fixtures only','external_fallback_cpu_graphs':len(external['cpu']),'external_fallback_runtime_runs':sum(len(v['runs']) for v in external['cpu'].values()),'external_fallback_boundaries':len(external['boundaries']),'element_codegen_cpu_runs':len(elements['cpu']),'element_codegen_eager_mismatches':['roundtrip_fp16'],'element_codegen_protocol_cases':len(elements['protocol']),'fusion_legality_cpu_runs':len(legality['cpu']),'fusion_legality_protocol_cases':len(legality['protocol']),'graph_context_cases':len(context),'wrapper_device_cases':len(device_wrapper),'backend_registration_workers':len(backend),'order_entry_cases':len(order),'constructor_metadata_cases':len(constructor),'priority_focus':bool(focus),'functional_wrapper_boundaries':len(wrapper),'reinplace_executed_cases':len(reinplace)-1,'triton_dependency':'unavailable','view_scatter_cases':len(scatter),'pattern_protocol_cases':len(pattern),'batch_communication_cases':len(batch),'incremental_paths':len(incremental),'cleanup_and_noop_cases':len(cleanup),'fx_cache_workers':5,'async_future':'pass','initial_properties_paths':6,'full_source_ranges':len(coverage),'fusion_cases':4,'storage_cases':len(storage),'special_cpu_execution_paths':5,'entry_boundary':'expected failure verified','rewrite_cases':len(rewrite),'finalization_and_interpretation':'pass','links':links,'nodes':len(nodes),'completed_module_articles':sum(n['status']=='本层完成' for n in nodes.values()),'completed_leaves':sum(n['status']=='叶子·逐行完成' for n in nodes.values()),'pending_queue':len(q),'next':state['next'],'stage_cases':len(r),'properties_experiment':'pass','scope':'引用和状态检查加已采集证据核对；不证明未讲源码已覆盖'}
(ROOT/'maintenance/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
