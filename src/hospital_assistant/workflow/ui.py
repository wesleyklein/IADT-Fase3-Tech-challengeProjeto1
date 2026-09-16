"""Interface local: iniciar, recuperar e revisar uma solicitação por vez."""
import streamlit as st
from hospital_assistant.workflow.service import WorkflowService

st.set_page_config(page_title='Hospital acadêmico', page_icon='📋', layout='centered')
st.title('Assistente hospitalar acadêmico')
st.caption('Pacientes e protocolos fictícios. Rascunhos sem validade clínica.')
with st.sidebar: 
    st.header("Tech Challenge 9IADT")
    st.subheader('Configuração para novas solicitações')
    config_path = st.text_input('Arquivo de configuração RAG', value='config/rag-v2.json')
    model_variant = st.selectbox('Modelo', ['base', 'adapted'],
        format_func=lambda value: 'Base' if value == 'base' else 'Com adaptador')
    prompt_variant = st.selectbox('Prompt', ['short', 'current'],
        format_func=lambda value: 'Curto' if value == 'short' else 'Atual')
    st.caption('Seleção inicial para avaliação: base + curto. Solicitações salvas mantêm seu conteúdo.')
selection = (config_path, model_variant, prompt_variant)
# Instância por sessão: não compartilha o modelo entre usuários do Streamlit.
if ('workflow_service' not in st.session_state
        or st.session_state.get('rag_selection') != selection):
    st.session_state.pop('workflow_service', None)
    st.session_state.workflow_service = WorkflowService(rag_config=config_path,
        rag_overrides={'model_variant': model_variant, 'prompt_variant': prompt_variant})
    st.session_state.rag_selection = selection
service = st.session_state.workflow_service
labels = {'Consultar protocolo': 'protocol', 'Consultar paciente': 'patient',
          'Consultar exames': 'exams', 'Rascunho de laudo': 'report',
          'Formulário de receita': 'prescription'}
# O formulário envia os campos juntos. A ação escolhida define quais são obrigatórios.
with st.form('request'):
    selected = st.selectbox('O que deseja fazer?', list(labels))
    patient = st.text_input('Identificador do paciente (para consultas e formulários)', placeholder='SYN-P001')
    question = st.text_area('Pergunta sobre protocolos')
    submitted = st.form_submit_button('Iniciar solicitação')
if submitted:
    try:
        with st.spinner('Preparando a solicitação…'):
            result = service.start(labels[selected], question, patient.strip())
        st.session_state.thread = result['thread_id']
    except Exception as exc:
        st.error(str(exc))
with st.sidebar:
    st.subheader('Retomar solicitação')
    existing = st.text_input('ID da solicitação')
    if st.button('Abrir'):
        try:
            service.show(existing.strip())
            st.session_state.thread = existing.strip()
        except Exception as exc:
            st.error(str(exc))
if st.session_state.get('thread'):
    try:
        result = service.show(st.session_state.thread)
        st.write('ID:', result['thread_id'])
        st.write('Situação:', result['status'])
        if result.get('error'):
            st.error(result['error'])
        if result.get('draft'):
            draft = result['draft']
            st.subheader('Conteúdo para revisão')
            if draft.get('content_missing'):
                st.warning('A resposta contém apenas referências ou não tem conteúdo. Rejeite o rascunho e revise a geração.')
            if 'answer' in draft:
                st.text(draft['answer'])
            st.json(draft)
        if result['status'] == 'awaiting_review':
            with st.form('review'):
                reviewer = st.text_input('Nome do revisor')
                comment = st.text_area('Observação da revisão')
                decision = st.radio('Decisão', ['Rejeitar', 'Aprovar rascunho didático'])
                confirm = st.form_submit_button('Registrar decisão')
            if confirm:
                service.review(result['thread_id'], 'reject' if decision == 'Rejeitar' else 'approve',
                               reviewer, result['draft_hash'], comment)
                st.rerun()
        if result.get('decision'):
            st.subheader('Decisão registrada')
            st.json(result['decision'])
        with st.expander('Histórico das etapas'):
            st.json(result['events'])
    except Exception as exc:
        st.error(str(exc))

