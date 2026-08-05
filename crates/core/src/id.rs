use uuid::Uuid;

pub fn new_id() -> Uuid {
    Uuid::new_v4()
}

pub fn new_session_id() -> super::types::SessionId {
    super::types::SessionId(new_id())
}

pub fn new_agent_id() -> super::types::AgentId {
    super::types::AgentId(new_id())
}

pub fn new_message_id() -> super::types::MessageId {
    super::types::MessageId(new_id())
}

pub fn new_user_id() -> super::types::UserId {
    super::types::UserId(new_id())
}

pub fn new_memory_id() -> super::types::MemoryId {
    super::types::MemoryId(new_id())
}

pub fn new_job_id() -> super::types::JobId {
    super::types::JobId(new_id())
}