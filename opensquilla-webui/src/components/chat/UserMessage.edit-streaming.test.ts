// @vitest-environment happy-dom

import { afterEach, describe, expect, it } from 'vitest'
import { createApp, nextTick } from 'vue'
import i18n from '@/i18n'
import type { ChatRenderedMessage } from '@/types/chat'
import UserMessage from './UserMessage.vue'

function makeMessage(): ChatRenderedMessage {
  return {
    id: 'edit-stream-msg',
    role: 'user',
    displayRole: 'user',
    roleLabel: 'You',
    text: 'edit me',
    timeStr: '',
    showHeader: false,
  }
}

async function renderUserMessage(isStreaming?: boolean) {
  const host = document.createElement('div')
  document.body.appendChild(host)
  const message = makeMessage()
  const app = createApp(UserMessage, {
    message,
    shareMode: false,
    shareSelected: false,
    shareMessageId: message.id,
    stripTimePrefix: (value: string) => value,
    copyMessage: async () => true,
    downloadAttachment: async () => true,
    isStreaming,
  })
  app.use(i18n)
  app.mount(host)
  await nextTick()
  return { app, host }
}

afterEach(() => {
  document.body.innerHTML = ''
  i18n.global.locale.value = 'en'
})

describe('UserMessage edit button streaming state', () => {
  it('disables edit button while streaming', async () => {
    const { app, host } = await renderUserMessage(true)

    const buttons = host.querySelectorAll<HTMLButtonElement>('.msg-action')
    const editButton = buttons[buttons.length - 1]
    expect(editButton).toBeTruthy()
    expect(editButton!.disabled).toBe(true)
    expect(editButton!.classList.contains('msg-action--disabled')).toBe(true)
    expect(editButton!.getAttribute('title')).toBe('Wait for the current reply to finish before editing')
    app.unmount()
  })

  it('keeps edit button enabled when idle', async () => {
    const { app, host } = await renderUserMessage(false)

    const buttons = host.querySelectorAll<HTMLButtonElement>('.msg-action')
    const editButton = buttons[buttons.length - 1]
    expect(editButton).toBeTruthy()
    expect(editButton!.disabled).toBe(false)
    expect(editButton!.classList.contains('msg-action--disabled')).toBe(false)
    expect(editButton!.getAttribute('title')).toBe('Edit')
    app.unmount()
  })
})
