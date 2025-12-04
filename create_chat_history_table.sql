-- ================================================================
-- CHAT HISTORY DATABASE SCHEMA FOR WEBSTORE CHATBOT
-- ================================================================
-- This script creates the necessary tables for persistent chat history storage
-- Run this in your MySQL database (eshop) to enable chat history persistence

USE eshop;

-- ================================================================
-- 1. MAIN CHAT CONVERSATIONS TABLE
-- ================================================================
-- Stores all chat messages with full context and metadata
CREATE TABLE IF NOT EXISTS chat_conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,           -- Unique session identifier
    user_email VARCHAR(255) NULL,               -- Customer email (if logged in)
    sender ENUM('user', 'bot', 'admin', 'admin_bot') NOT NULL,  -- Who sent the message
    message TEXT NOT NULL,                      -- The actual message content
    intent VARCHAR(100) NULL,                   -- Detected intent (order_status, product_search, etc.)
    confidence FLOAT NULL,                      -- AI confidence score (0.0 to 1.0)
    current_page VARCHAR(255) NULL,             -- Page user was on when sending message
    user_agent TEXT NULL,                       -- Browser/device information
    ip_address VARCHAR(45) NULL,                -- User's IP address for analytics
    response_time_ms INT NULL,                  -- Time taken to generate response
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When message was sent
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for better query performance
    INDEX idx_session_id (session_id),
    INDEX idx_user_email (user_email),
    INDEX idx_created_at (created_at),
    INDEX idx_intent (intent),
    INDEX idx_sender (sender)
);

-- ================================================================
-- 2. CHAT SESSIONS TABLE
-- ================================================================
-- Tracks chat sessions with summary information
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL UNIQUE,   -- Unique session identifier
    user_email VARCHAR(255) NULL,               -- Customer email (if logged in)
    customer_id INT NULL,                       -- Link to customer table if exists
    session_type ENUM('customer', 'admin') DEFAULT 'customer',  -- Type of chat session
    first_message_at TIMESTAMP NULL,            -- When session started
    last_message_at TIMESTAMP NULL,             -- When session ended/last activity
    total_messages INT DEFAULT 0,               -- Total messages in session
    total_user_messages INT DEFAULT 0,          -- Messages sent by user
    total_bot_messages INT DEFAULT 0,           -- Messages sent by bot
    average_response_time_ms FLOAT NULL,        -- Average bot response time
    session_duration_seconds INT NULL,          -- Total session duration
    primary_intent VARCHAR(100) NULL,           -- Most common intent in session
    customer_satisfaction ENUM('positive', 'neutral', 'negative') NULL,  -- If we add rating
    resolution_status ENUM('resolved', 'unresolved', 'escalated') NULL,  -- Issue resolution
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key constraint if customer table exists
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id) ON DELETE SET NULL,
    
    -- Indexes for analytics queries
    INDEX idx_user_email (user_email),
    INDEX idx_session_type (session_type),
    INDEX idx_created_at (created_at),
    INDEX idx_primary_intent (primary_intent)
);

-- ================================================================
-- 3. CHAT ANALYTICS TABLE
-- ================================================================
-- Stores daily/hourly analytics for business intelligence
CREATE TABLE IF NOT EXISTS chat_analytics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date_recorded DATE NOT NULL,                -- Date of analytics (YYYY-MM-DD)
    hour_recorded TINYINT NULL,                 -- Hour of day (0-23) for hourly analytics
    total_sessions INT DEFAULT 0,               -- Total chat sessions
    total_messages INT DEFAULT 0,               -- Total messages exchanged
    unique_users INT DEFAULT 0,                 -- Unique users who chatted
    average_session_duration FLOAT NULL,        -- Average session length in seconds
    average_response_time FLOAT NULL,           -- Average bot response time
    most_common_intent VARCHAR(100) NULL,       -- Most frequent intent
    resolved_sessions INT DEFAULT 0,            -- Sessions marked as resolved
    escalated_sessions INT DEFAULT 0,           -- Sessions that needed escalation
    customer_satisfaction_score FLOAT NULL,     -- Average satisfaction (if implemented)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique entries per date/hour
    UNIQUE KEY unique_date_hour (date_recorded, hour_recorded),
    INDEX idx_date (date_recorded)
);

-- ================================================================
-- 4. COMMON QUERIES TABLE (OPTIONAL)
-- ================================================================
-- Stores frequently asked questions for knowledge base improvement
CREATE TABLE IF NOT EXISTS chat_common_queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_text TEXT NOT NULL,                   -- The user's question/message
    normalized_query TEXT NOT NULL,             -- Cleaned/normalized version
    frequency_count INT DEFAULT 1,              -- How many times this was asked
    intent VARCHAR(100) NULL,                   -- Associated intent
    average_confidence FLOAT NULL,              -- Average AI confidence for this query
    suggested_response TEXT NULL,               -- Recommended response (for training)
    last_asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_intent (intent),
    INDEX idx_frequency (frequency_count DESC),
    INDEX idx_last_asked (last_asked_at)
);

-- ================================================================
-- 5. SAMPLE DATA VIEWS FOR ANALYTICS
-- ================================================================

-- View: Recent Chat Activity (Last 24 hours)
CREATE OR REPLACE VIEW recent_chat_activity AS
SELECT 
    DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00') as hour_bucket,
    COUNT(*) as message_count,
    COUNT(DISTINCT session_id) as session_count,
    COUNT(DISTINCT user_email) as unique_users,
    AVG(response_time_ms) as avg_response_time
FROM chat_conversations 
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY hour_bucket
ORDER BY hour_bucket DESC;

-- View: Popular Intents Today
CREATE OR REPLACE VIEW todays_popular_intents AS
SELECT 
    intent,
    COUNT(*) as frequency,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT session_id) as unique_sessions
FROM chat_conversations 
WHERE DATE(created_at) = CURDATE() 
AND intent IS NOT NULL
GROUP BY intent
ORDER BY frequency DESC;

-- View: Admin Dashboard Summary
CREATE OR REPLACE VIEW admin_chat_summary AS
SELECT 
    DATE(created_at) as chat_date,
    COUNT(DISTINCT session_id) as total_sessions,
    COUNT(*) as total_messages,
    COUNT(DISTINCT user_email) as unique_customers,
    AVG(response_time_ms) as avg_response_time_ms,
    SUM(CASE WHEN sender = 'user' THEN 1 ELSE 0 END) as user_messages,
    SUM(CASE WHEN sender = 'bot' THEN 1 ELSE 0 END) as bot_messages
FROM chat_conversations 
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(created_at)
ORDER BY chat_date DESC;

-- ================================================================
-- 6. HELPFUL INDEXES FOR PERFORMANCE
-- ================================================================

-- Composite indexes for common query patterns
ALTER TABLE chat_conversations 
ADD INDEX idx_session_created (session_id, created_at),
ADD INDEX idx_user_date (user_email, created_at),
ADD INDEX idx_intent_date (intent, created_at);

-- ================================================================
-- 7. CLEANUP PROCEDURES (OPTIONAL)
-- ================================================================

-- Procedure to clean old chat data (keep last 6 months)
DELIMITER //
CREATE PROCEDURE CleanOldChatData()
BEGIN
    -- Delete conversations older than 6 months
    DELETE FROM chat_conversations 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL 6 MONTH);
    
    -- Delete old analytics data older than 1 year
    DELETE FROM chat_analytics 
    WHERE date_recorded < DATE_SUB(CURDATE(), INTERVAL 1 YEAR);
    
    -- Delete unused sessions
    DELETE FROM chat_sessions 
    WHERE session_id NOT IN (SELECT DISTINCT session_id FROM chat_conversations);
END //
DELIMITER ;

-- ================================================================
-- SETUP COMPLETE!
-- ================================================================
-- You can now run the following to verify tables were created:
-- SHOW TABLES LIKE 'chat_%';
-- DESCRIBE chat_conversations;
-- DESCRIBE chat_sessions;
-- DESCRIBE chat_analytics;