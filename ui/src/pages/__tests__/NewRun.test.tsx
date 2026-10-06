import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import NewRun from '../NewRun'
import { renderWithProviders } from '../../test/utils'
import * as api from '../../api'

vi.mock('../../api', async () => {
  const actual = await vi.importActual('../../api')
  return { ...actual, createRun: vi.fn() }
})

describe('NewRun', () => {
  const mockCreateRun = vi.mocked(api.createRun)

  beforeEach(() => {
    vi.clearAllMocks()
    mockCreateRun.mockResolvedValue({ run_id: 'run-1', status: 'PENDING', message: 'queued' })
  })

  it('uses the configured provider default unless a model override is selected', async () => {
    const user = userEvent.setup()
    renderWithProviders(<NewRun />)

    expect(screen.getByLabelText(/llm model/i)).toHaveValue('')
    await user.type(
      screen.getByLabelText(/research question/i),
      'Does treatment improve the primary clinical outcome?',
    )
    await user.click(screen.getByRole('button', { name: /start research run/i }))

    expect(mockCreateRun.mock.calls[0][0]).toEqual({
      research_question: 'Does treatment improve the primary clinical outcome?',
      max_results: 100,
      output_formats: ['markdown', 'json'],
      dod_level: 1,
    })
  })

  it('passes a provider-compatible model override when entered', async () => {
    const user = userEvent.setup()
    renderWithProviders(<NewRun />)

    await user.type(
      screen.getByLabelText(/research question/i),
      'Does treatment improve the primary clinical outcome?',
    )
    await user.type(screen.getByLabelText(/llm model override/i), 'gemini-3.8-flash')
    await user.click(screen.getByRole('button', { name: /start research run/i }))

    expect(mockCreateRun.mock.calls[0][0]).toMatchObject({
      model: 'gemini-3.8-flash',
    })
  })
})
